"""
FastAPI WebSocket server for streaming Hindi ASR.

Protocol:
  Client -> Server:
    Binary:  raw PCM audio (s16le, 16kHz mono)
    Text:    {"type": "flush"}   — end of utterance, return final transcript
    Text:    {"type": "end"}     — close session

  Server -> Client:
    Text:    {"type": "interim", "text": "...", "confidence": 0.9}
    Text:    {"type": "final",   "text": "...", "confidence": 0.95}
    Text:    {"type": "error",   "message": "..."}

Usage:
    python server.py
    python server.py --host 0.0.0.0 --port 8000 --latency-ms 1
"""

import argparse
import asyncio
import json
import logging
import os
import uuid
from contextlib import asynccontextmanager
from typing import Any

from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware

from asr_engine import ASREngine

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger("stt-server")

# ---------------------------------------------------------------------------
# Defaults (overridable via CLI / env vars)
# ---------------------------------------------------------------------------
DEFAULT_HOST = "0.0.0.0"
DEFAULT_PORT = 8000
DEFAULT_MODEL = "salesken/Hindi-FastConformer-Streaming-ASR"
DEFAULT_DEVICE = "cuda"
DEFAULT_LATENCY_MS = 1       # 80ms right context
DEFAULT_DECODER = "rnnt"      # "rnnt" or "ctc"
DEFAULT_CHUNK_MS = 320        # process audio in 320ms chunks
SAMPLE_RATE = 16000

# ---------------------------------------------------------------------------
# CLI args
# ---------------------------------------------------------------------------
def parse_args():
    p = argparse.ArgumentParser(description="Hindi Streaming ASR Server")
    p.add_argument("--host", default=os.getenv("STT_HOST", DEFAULT_HOST))
    p.add_argument("--port", type=int, default=int(os.getenv("STT_PORT", str(DEFAULT_PORT))))
    p.add_argument("--model", default=os.getenv("STT_MODEL", DEFAULT_MODEL))
    p.add_argument("--model-path", default=os.getenv("STT_MODEL_PATH"), help="Path to local .nemo file (overrides --model)")
    p.add_argument("--device", default=os.getenv("STT_DEVICE", DEFAULT_DEVICE))
    p.add_argument("--latency-ms", type=int, default=int(os.getenv("STT_LATENCY_MS", str(DEFAULT_LATENCY_MS))))
    p.add_argument("--decoder", default=os.getenv("STT_DECODER", DEFAULT_DECODER))
    p.add_argument("--chunk-ms", type=int, default=int(os.getenv("STT_CHUNK_MS", str(DEFAULT_CHUNK_MS))))
    return p.parse_args()


# ---------------------------------------------------------------------------
# Session — per-connection state
# ---------------------------------------------------------------------------
class Session:
    """Manages audio buffering and model state for one WebSocket connection."""

    def __init__(self, engine: ASREngine, chunk_bytes: int):
        self.id: str = uuid.uuid4().hex[:8]
        self.engine = engine
        self.state = engine.create_state()
        self.buffer = bytearray()
        self.chunk_bytes = chunk_bytes   # bytes per processing chunk (int16)
        self.step_num = 0

    def feed(self, data: bytes) -> None:
        self.buffer.extend(data)

    def has_chunk(self) -> bool:
        return len(self.buffer) >= self.chunk_bytes

    async def process_chunks(self) -> list[str]:
        """Drain complete chunks from the buffer, return interim transcriptions."""
        results: list[str] = []
        while self.has_chunk():
            chunk = bytes(self.buffer[: self.chunk_bytes])
            self.buffer = self.buffer[self.chunk_bytes :]
            text, self.state = await asyncio.to_thread(
                self.engine.process_chunk, chunk, self.state
            )
            self.step_num += 1
            if text:
                results.append(text)
        return results

    async def flush(self) -> str:
        """Process remaining buffer audio as the final chunk."""
        if not self.buffer:
            return ""
        remaining = bytes(self.buffer)
        self.buffer.clear()
        text = await asyncio.to_thread(
            self.engine.process_final, remaining, self.state
        )
        return text

    def close(self) -> None:
        self.buffer.clear()


# ---------------------------------------------------------------------------
# FastAPI app
# ---------------------------------------------------------------------------
def create_app(engine: ASREngine, chunk_ms: int) -> FastAPI:
    chunk_bytes = int(chunk_ms * SAMPLE_RATE / 1000) * 2  # 2 bytes per int16 sample

    @asynccontextmanager
    async def lifespan(app: FastAPI):
        logger.info("ASR engine ready (chunk=%dms, %d bytes)", chunk_ms, chunk_bytes)
        yield
        logger.info("Shutting down")

    app = FastAPI(title="Hindi Streaming ASR", lifespan=lifespan)
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_methods=["*"],
        allow_headers=["*"],
    )

    sessions: dict[str, Session] = {}

    @app.websocket("/ws")
    async def ws_endpoint(websocket: WebSocket):
        await websocket.accept()

        session = Session(engine, chunk_bytes)
        sessions[session.id] = session
        logger.info("[%s] connected  (sessions=%d)", session.id, len(sessions))

        try:
            while True:
                message = await websocket.receive()

                # Client disconnected
                if message["type"] == "websocket.disconnect":
                    break

                # --- Text (control) messages ---
                if "text" in message:
                    try:
                        data: dict[str, Any] = json.loads(message["text"])
                    except json.JSONDecodeError:
                        continue

                    msg_type = data.get("type")

                    if msg_type == "flush":
                        final_text = await session.flush()
                        await websocket.send_text(
                            json.dumps({"type": "final", "text": final_text, "confidence": 0.95})
                        )

                    elif msg_type == "end":
                        final_text = await session.flush()
                        await websocket.send_text(
                            json.dumps({"type": "final", "text": final_text, "confidence": 0.95})
                        )
                        break

                # --- Binary (audio) messages ---
                elif "bytes" in message:
                    session.feed(message["bytes"])

                    # Process all complete chunks
                    interim_texts = await session.process_chunks()
                    for text in interim_texts:
                        await websocket.send_text(
                            json.dumps({"type": "interim", "text": text, "confidence": 0.9})
                        )

        except WebSocketDisconnect:
            logger.info("[%s] disconnected", session.id)
        except Exception:
            logger.exception("[%s] error", session.id)
        finally:
            session.close()
            sessions.pop(session.id, None)
            logger.info("[%s] session cleaned up  (sessions=%d)", session.id, len(sessions))

    @app.get("/health")
    async def health():
        return {
            "status": "ok",
            "model": engine.model._model_name if hasattr(engine.model, "_model_name") else "loaded",
            "device": str(engine.device),
            "active_sessions": len(sessions),
        }

    return app


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------
def main():
    args = parse_args()

    logger.info("Starting ASR server with config:")
    logger.info("  model=%s  device=%s  latency=%dms  decoder=%s  chunk=%dms",
                args.model, args.device, args.latency_ms, args.decoder, args.chunk_ms)

    engine = ASREngine(
        model_name=args.model,
        model_path=args.model_path,
        device=args.device,
        latency_ms=args.latency_ms,
        decoder_type=args.decoder,
    )

    app = create_app(engine, args.chunk_ms)

    import uvicorn
    uvicorn.run(app, host=args.host, port=args.port, log_level="info")


if __name__ == "__main__":
    main()
