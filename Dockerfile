ARG DEVICE=cpu

# ==============================================================================
# Python dependency builders (wheels only, matching runtime Python ABI)
# ==============================================================================
FROM python:3.13-slim AS pydeps-py313
ARG DEVICE=cpu
ENV PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1

# Only packages the client actually imports are installed. --no-deps keeps the
# heavy transitive stacks (torch, transformers, numba, scipy...) out entirely.
RUN case "$DEVICE" in \
        openvino) ORT=onnxruntime-openvino==1.24.1 ;; \
        *) ORT=onnxruntime==1.24.4 ;; \
    esac && \
    pip install --no-compile --no-deps --prefix=/install \
        numpy \
        requests \
        urllib3 \
        certifi \
        idna \
        charset-normalizer \
        soundfile \
        typing_extensions \
        soxr \
        cffi \
        pycparser \
        "$ORT" && \
    rm -rf \
        /install/lib/python3.13/site-packages/*/tests \
        /install/lib/python3.13/site-packages/numpy/f2py \
        /install/lib/python3.13/site-packages/numpy/tests \
        /install/lib/python3.13/site-packages/onnxruntime/tools \
        /install/lib/python3.13/site-packages/onnxruntime/transformers \
        /install/lib/python3.13/site-packages/onnxruntime/quantization \
        /install/lib/python3.13/site-packages/onnxruntime/datasets && \
    if [ "$DEVICE" = "openvino" ]; then \
        rm -rf \
            /install/lib/python3.13/site-packages/onnxruntime/capi/libopenvino_intel_npu_plugin.so*; \
    fi && \
    find /install -type d -name '__pycache__' -prune -exec rm -rf {} + 2>/dev/null || true

FROM python:3.12-slim AS pydeps-py312
ENV PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1
RUN pip install --no-compile --no-deps --prefix=/install \
        numpy \
        requests \
        urllib3 \
        certifi \
        idna \
        charset-normalizer \
        soundfile \
        typing_extensions \
        soxr \
        cffi \
        pycparser \
        onnxruntime-gpu==1.32.0 && \
    rm -rf \
        /install/lib/python3.12/site-packages/*/tests \
        /install/lib/python3.12/site-packages/numpy/f2py \
        /install/lib/python3.12/site-packages/numpy/tests \
        /install/lib/python3.12/site-packages/onnxruntime/tools \
        /install/lib/python3.12/site-packages/onnxruntime/transformers \
        /install/lib/python3.12/site-packages/onnxruntime/quantization \
        /install/lib/python3.12/site-packages/onnxruntime/datasets && \
    find /install -type d -name '__pycache__' -prune -exec rm -rf {} + 2>/dev/null || true

# ==============================================================================
# CPU runtime (distroless: no shell, apt, pip or compilers)
# ==============================================================================
FROM gcr.io/distroless/python3-debian13:latest AS final-cpu
ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PYTHONPATH=/app:/opt/py
WORKDIR /app
COPY --from=pydeps-py313 /install/lib/python3.13/site-packages /opt/py
COPY absmouth /app/absmouth
ENTRYPOINT ["python3", "-m", "absmouth.subsonic_client"]

# ==============================================================================
# OpenVINO runtime (Intel GPU / OpenCL)
# ==============================================================================
FROM ubuntu:24.04 AS ovdeps
RUN apt-get update && \
    apt-get install -y --no-install-recommends --download-only intel-opencl-icd && \
    mkdir -p /ov && \
    for deb in /var/cache/apt/archives/*.deb; do dpkg-deb -x "$deb" /ov; done && \
    sed -i 's#/usr/lib/x86_64-linux-gnu/intel-opencl/libigdrcl.so#/opt/ov/lib/intel-opencl/libigdrcl.so#' \
        /ov/etc/OpenCL/vendors/intel.icd && \
    rm -f /ov/usr/lib/x86_64-linux-gnu/libicu*.so*

FROM gcr.io/distroless/python3-debian13:latest AS final-openvino
ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PYTHONPATH=/app:/opt/py \
    LD_LIBRARY_PATH=/opt/ov/lib:/opt/ov/lib/intel-opencl
WORKDIR /app
COPY --from=ovdeps /ov/usr/lib/x86_64-linux-gnu /opt/ov/lib
COPY --from=ovdeps /ov/etc/OpenCL /etc/OpenCL
COPY --from=pydeps-py313 /install/lib/python3.13/site-packages /opt/py
COPY absmouth /app/absmouth
ENTRYPOINT ["python3", "-m", "absmouth.subsonic_client"]

# ==============================================================================
# CUDA runtime
# ==============================================================================
FROM ubuntu:24.04 AS final-cuda
ENV DEBIAN_FRONTEND=noninteractive \
    PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PYTHONPATH=/app:/opt/py \
    NVIDIA_VISIBLE_DEVICES=all \
    NVIDIA_DRIVER_CAPABILITIES=compute,utility \
    LD_LIBRARY_PATH=/usr/local/nvidia/lib64:/usr/local/cuda/lib64
WORKDIR /app

RUN apt-get update && \
    apt-get install -y --no-install-recommends ca-certificates wget && \
    wget -q https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2404/x86_64/cuda-keyring_1.1-1_all.deb -O /tmp/cuda-keyring.deb && \
    dpkg -i /tmp/cuda-keyring.deb && \
    rm -f /tmp/cuda-keyring.deb && \
    apt-get update && \
    apt-get install -y --no-install-recommends \
        python3 \
        cuda-cudart-13-0 \
        libcublas-13-0 \
        libcurand-13-0 \
        libcudnn9-cuda-13=9.18.0.77-1 && \
    apt-get purge -y wget && \
    apt-get autoremove -y && \
    rm -rf \
        /var/lib/apt/lists/* \
        /var/cache/apt/* \
        /usr/share/doc/* \
        /usr/share/man/* \
        /usr/lib/python3/dist-packages/* \
        /usr/local/cuda-13.0/compat \
        /usr/lib/x86_64-linux-gnu/libcudnn_adv.so* \
        /usr/lib/x86_64-linux-gnu/libcudnn_ext.so*

COPY --from=pydeps-py312 /install/lib/python3.12/site-packages /opt/py
COPY absmouth /app/absmouth
ENTRYPOINT ["python3", "-m", "absmouth.subsonic_client"]

# ==============================================================================
# Selected runtime
# ==============================================================================
FROM final-${DEVICE} AS final
