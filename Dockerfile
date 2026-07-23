ARG DEVICE=cpu

# ==============================================================================
# Production Device Base Stages
# ==============================================================================

# CPU Base
FROM python:3.13-slim AS prod-cpu
ENV DEVICE=cpu

# OpenVINO (Intel GPU) Base
FROM python:3.13-slim AS prod-openvino
ENV DEVICE=openvino \
    OV_FRONTEND_CACHE_ENABLE=1

RUN apt-get update && \
    apt-get install -y --no-install-recommends \
    clinfo \
    ocl-icd-libopencl1 \
    wget && \
    wget -nv https://github.com/intel/intel-graphics-compiler/releases/download/v2.36.3/intel-igc-core-2_2.36.3+21719_amd64.deb && \
    wget -nv https://github.com/intel/intel-graphics-compiler/releases/download/v2.36.3/intel-igc-opencl-2_2.36.3+21719_amd64.deb && \
    wget -nv https://github.com/intel/compute-runtime/releases/download/26.22.38646.4/intel-opencl-icd_26.22.38646.4-0_amd64.deb && \
    wget -nv https://github.com/intel/intel-graphics-compiler/releases/download/igc-1.0.17537.24/intel-igc-core_1.0.17537.24_amd64.deb && \
    wget -nv https://github.com/intel/intel-graphics-compiler/releases/download/igc-1.0.17537.24/intel-igc-opencl_1.0.17537.24_amd64.deb && \
    wget -nv https://github.com/intel/compute-runtime/releases/download/24.35.30872.36/intel-opencl-icd-legacy1_24.35.30872.36_amd64.deb && \
    wget -nv https://github.com/intel/compute-runtime/releases/download/26.22.38646.4/libigdgmm12_22.10.0_amd64.deb && \
    apt-get install -y --no-install-recommends ./*.deb && \
    rm *.deb && \
    apt-get remove -yqq wget && \
    apt-get autoremove -yqq && \
    rm -rf /var/lib/apt/lists/*

# CUDA (NVIDIA GPU) Base
FROM nvidia/cuda:13.0.0-runtime-ubuntu24.04 AS prod-cuda
ENV DEVICE=cuda \
    NVIDIA_VISIBLE_DEVICES=all \
    NVIDIA_DRIVER_CAPABILITIES=compute,utility \
    LD_LIBRARY_PATH=/usr/local/nvidia/lib64:${LD_LIBRARY_PATH}

RUN apt-get update && \
    apt-get install -y --no-install-recommends \
    libcudnn9-cuda-13 && \
    rm -rf /var/lib/apt/lists/*


# ==============================================================================
# Final Runtime Stage
# ==============================================================================
FROM prod-${DEVICE} AS final

ARG DEVICE=cpu
ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PATH="/opt/venv/bin:$PATH"

WORKDIR /app

# Install Python 3, venv, and audio decoding system packages
RUN apt-get update && \
    apt-get install -y --no-install-recommends \
    python3 \
    python3-pip \
    python3-venv \
    ffmpeg \
    libsndfile1 && \
    rm -rf /var/lib/apt/lists/*

RUN python3 -m venv /opt/venv

RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir \
    numpy \
    protobuf \
    packaging \
    flatbuffers

# Install PyTorch CPU wheels
RUN pip install \
    --index-url https://download.pytorch.org/whl/cpu \
    torch \
    torchaudio \
    torchcodec

# Install machine learning & audio processing dependencies
RUN pip install --no-cache-dir \
    numpy \
    einops \
    transformers \
    onnx \
    onnxscript \
    requests \
    librosa

# Install ONNX Runtime variant corresponding to target DEVICE
RUN if [ "$DEVICE" = "cuda" ]; then \
        pip install --no-cache-dir --no-deps onnxruntime-gpu; \
    else \
        pip install --no-cache-dir --no-deps onnxruntime; \
    fi

COPY pyproject.toml README.md ./
COPY absmouth/ ./absmouth/

# Use --no-deps so pyproject.toml's CPU 'onnxruntime' requirement does not overwrite onnxruntime-gpu
RUN pip install --no-cache-dir --no-deps .

ENTRYPOINT ["absmouth"]
