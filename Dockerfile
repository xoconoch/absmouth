# Use an official Python image
FROM python:3.13-slim

# Prevent Python from buffering stdout/stderr
ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1

WORKDIR /app

# Install system dependencies (ffmpeg and libsndfile1 for audio decoding)
RUN apt-get update && \
    apt-get install -y --no-install-recommends ffmpeg libsndfile1 && \
    rm -rf /var/lib/apt/lists/*

# Install python dependencies matching the template
RUN pip install --no-cache-dir \
    numpy \
    protobuf \
    packaging \
    flatbuffers

RUN pip install --no-cache-dir --no-deps onnxruntime

RUN pip install \
    --index-url https://download.pytorch.org/whl/cpu \
    torch \
    torchaudio \
    torchcodec

RUN pip install --no-cache-dir \
    numpy \
    einops \
    transformers \
    onnx \
    onnxscript \
    requests \
    librosa

# Copy Python package files and install
COPY pyproject.toml README.md ./
COPY absmouth/ ./absmouth/

RUN pip install --no-cache-dir .

ENTRYPOINT ["absmouth"]
