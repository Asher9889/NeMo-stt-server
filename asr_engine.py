"""
NeMo-based streaming ASR engine for Hindi FastConformer.

Wraps the salesken/Hindi-FastConformer-Streaming-ASR model for real-time
inference with per-connection cache state.
"""

import glob as glob_mod
import os

import numpy as np
import torch
import nemo.collections.asr as nemo_asr
from dataclasses import dataclass
from typing import Any
from logging import getLogger

logger = getLogger(__name__)

# Attention context sizes and their corresponding latency in ms
# [left_context, right_context] -> latency_ms
LATENCY_PRESETS: dict[int, list[int]] = {
    0: [70, 0],   # ~0ms worst-case latency
    1: [70, 1],   # ~80ms
    16: [70, 16],  # ~480ms
    33: [70, 33],  # ~1040ms (default)
}


@dataclass
class StreamState:
    """Per-connection streaming state. Holds cache tensors and decoder state."""

    cache_last_channel: Any = None
    cache_last_time: Any = None
    cache_last_channel_len: Any = None
    previous_hypotheses: Any = None
    previous_pred_out: Any = None
    drop_extra_pre_encoded: int = 0
    step_num: int = 0


class ASREngine:
    """
    Wraps a NeMo streaming ASR model for real-time chunk-based inference.

    Usage:
        engine = ASREngine(model_name="salesken/Hindi-FastConformer-Streaming-ASR")
        state = engine.create_state()
        text, state = engine.process_chunk(audio_bytes, state)
        final_text = engine.process_final(remaining_bytes, state)
    """

    def __init__(
        self,
        model_name: str = "salesken/Hindi-FastConformer-Streaming-ASR",
        model_path: str | None = None,
        device: str = "cuda",
        latency_ms: int = 1,
        decoder_type: str = "rnnt",
    ):
        self.device = torch.device(device)
        self.sample_rate = 16000
        self.model = self._load_model(model_name, model_path, latency_ms, decoder_type)

    def _find_nemo_file(self, model_name: str) -> str:
        """Locate the .nemo file in NeMo's HuggingFace cache."""
        cache_root = os.path.expanduser("~/.cache/torch/NeMo")
        pattern = os.path.join(cache_root, "**", model_name, "**", "*.nemo")
        matches = sorted(
            glob_mod.glob(pattern, recursive=True), key=os.path.getmtime, reverse=True
        )
        if matches:
            logger.info("Found .nemo file in cache: %s", matches[0])
            return matches[0]
        raise FileNotFoundError(
            f"No .nemo file for '{model_name}' found in {cache_root}. "
            "Run once to download, or pass --model-path to a local .nemo file."
        )

    def _load_model(self, model_name: str, model_path: str | None, latency_ms: int, decoder_type: str):
        logger.info("Loading model: %s", model_name)

        # Determine the .nemo file to load
        if model_path:
            nemo_path = model_path
        else:
            nemo_path = self._find_nemo_file(model_name)

        logger.info("Restoring from: %s", nemo_path)
        model = nemo_asr.models.EncDecHybridRNNTCTCBPEModel.restore_from(nemo_path)

        # Set streaming latency via attention context size
        att_context = LATENCY_PRESETS.get(latency_ms, LATENCY_PRESETS[1])
        model.encoder.set_default_att_context_size(att_context)
        logger.info("Attention context size: %s (latency ~%dms)", att_context, latency_ms * 80)

        # Set decoder type (rnnt or ctc)
        model.change_decoding_strategy(decoder_type=decoder_type)
        logger.info("Decoder type: %s", decoder_type)

        model.eval()
        model.to(self.device)

        logger.info("Model loaded on %s", self.device)
        return model

    def create_state(self) -> StreamState:
        """Create initial streaming state for a new connection."""
        cache_ch, cache_time, cache_len = self.model.encoder.get_initial_cache_state(
            batch_size=1
        )
        return StreamState(
            cache_last_channel=cache_ch,
            cache_last_time=cache_time,
            cache_last_channel_len=cache_len,
        )

    def _resolve_drop_extra(self, state: StreamState) -> int:
        """Drop extra pre-encoded tokens: 0 on first step, model value after."""
        if state.step_num == 0:
            return 0
        return self.model.encoder.streaming_cfg.drop_extra_pre_encoded

    @torch.inference_mode()
    def _infer(self, tensor: torch.Tensor, state: StreamState, keep_all: bool):
        """Run one streaming step through the model (preprocess → encoder → decoder)."""
        audio_len = torch.tensor([tensor.shape[1]], device=self.device)

        # Step 1: Preprocessor — raw PCM → mel-spectrogram features (batch, dim, time)
        processed_signal, processed_signal_length = self.model.preprocessor(
            input_signal=tensor,
            length=audio_len,
        )

        # Step 2: Conformer stream step — encoder + decoder with cache
        (
            pred_out,
            transcribed_texts,
            cache_ch,
            cache_time,
            cache_len,
            hyps,
        ) = self.model.conformer_stream_step(
            processed_signal=processed_signal,
            processed_signal_length=processed_signal_length,
            cache_last_channel=state.cache_last_channel,
            cache_last_time=state.cache_last_time,
            cache_last_channel_len=state.cache_last_channel_len,
            keep_all_outputs=keep_all,
            previous_hypotheses=state.previous_hypotheses,
            previous_pred_out=state.previous_pred_out,
            drop_extra_pre_encoded=self._resolve_drop_extra(state),
            return_transcription=True,
        )

        return transcribed_texts, pred_out, cache_ch, cache_time, cache_len, hyps

    def _bytes_to_tensor(self, audio_bytes: bytes) -> torch.Tensor:
        """Convert int16 PCM bytes to float32 tensor of shape (1, num_samples)."""
        audio = np.frombuffer(audio_bytes, dtype=np.int16).astype(np.float32) / 32768.0
        return torch.tensor(audio, device=self.device).unsqueeze(0)

    def _extract_text(self, transcribed_texts) -> str:
        """Extract plain text from model output."""
        if not transcribed_texts:
            return ""
        text = transcribed_texts[0]
        if hasattr(text, "text"):
            return text.text
        return str(text)

    def _update_state(self, state: StreamState, pred_out, cache_ch, cache_time, cache_len, hyps) -> StreamState:
        """Create updated state from inference results."""
        return StreamState(
            cache_last_channel=cache_ch,
            cache_last_time=cache_time,
            cache_last_channel_len=cache_len,
            previous_hypotheses=hyps,
            previous_pred_out=pred_out,
            drop_extra_pre_encoded=self.model.encoder.streaming_cfg.drop_extra_pre_encoded,
            step_num=state.step_num + 1,
        )

    def process_chunk(self, audio_bytes: bytes, state: StreamState) -> tuple[str, StreamState]:
        """
        Process one audio chunk through the model.

        Args:
            audio_bytes: Raw int16 PCM bytes (16kHz mono).
            state: Current streaming state.

        Returns:
            (transcription_text, updated_state)
        """
        tensor = self._bytes_to_tensor(audio_bytes)
        texts, pred_out, cache_ch, cache_time, cache_len, hyps = self._infer(
            tensor, state, keep_all=False
        )
        text = self._extract_text(texts)
        new_state = self._update_state(state, pred_out, cache_ch, cache_time, cache_len, hyps)
        return text, new_state

    def process_final(self, audio_bytes: bytes, state: StreamState) -> str:
        """
        Process final audio chunk with keep_all_outputs=True.

        Call this when the utterance ends to get the complete transcription.
        """
        if not audio_bytes:
            return ""
        tensor = self._bytes_to_tensor(audio_bytes)
        texts, _, _, _, _, _ = self._infer(tensor, state, keep_all=True)
        return self._extract_text(texts)
