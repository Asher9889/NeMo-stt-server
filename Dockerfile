FROM nvidia/cuda:12.4.1-cudnn-runtime-ubuntu22.04

WORKDIR /app

# System deps for NeMo / torchaudio
RUN apt-get update && apt-get install -y --no-install-recommends \
    python3 python3-pip python3-dev \
    libsndfile1 ffmpeg \
    && rm -rf /var/lib/apt/lists/*

# Python deps
COPY requirements.txt .
RUN pip3 install --no-cache-dir -r requirements.txt

# Pre-download the model at build time (optional, avoids cold start)
# Uncomment the next line to bake the model into the image:
# RUN python3 -c "import nemo.collections.asr as nemo_asr; nemo_asr.models.EncDecHybridRNNTCTCBPEModel.from_pretrained('salesken/Hindi-FastConformer-Streaming-ASR')"

COPY . .

EXPOSE 8000

CMD ["python3", "server.py"]
