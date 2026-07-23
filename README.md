# Absolute Pitch Subsonic Sync Client (absmouth)

`absmouth` is an [abspitch](https://github.com/xoconoch/abspitch) client that
recursively crawls a Subsonic/OpenSubsonic server library, processes audio
tracks in 10-second chunks, generates 1024-dimensional acoustic embeddings, and
pushes them to the API.

---

## 1. Prerequisites & Model Setup

The only supported way to run `absmouth` is via **Docker**.

This tool requires the `muq_v0.0.1.onnx` acoustic embedding model (~1.2 GB).
Support for other models should be possible as long as they generate embeddings
of the same dimension, but I am haven't explored that possibility.

### Automatic or Manual Model Setup
By default, **the client automatically downloads the model** if it is not detected at the configured path inside the mounted data volume.

If you prefer to download it manually:
* **ONNX Model URL:** https://github.com/xoconoch/muq_onnx/releases/download/v0.0.1/muq_v0.0.1.onnx

Place the downloaded file (`muq_v0.0.1.onnx`) inside your host's `./data` directory (so that it is available inside the container as `data/muq_v0.0.1.onnx`).

---

## 2. Configuration

Configure the application by creating a `.env` file in your working directory:

```env
# Subsonic Connection Parameters
SUBSONIC_URL="http://localhost:4533"
SUBSONIC_USER="your_subsonic_username"
SUBSONIC_PASS="your_subsonic_password"

# Absolute Pitch Target API Endpoint
API_URL="http://localhost:8100/tracks"

# ONNX Model Path (defaults to 'data/muq_v0.0.1.onnx')
MODEL_PATH="data/muq_v0.0.1.onnx"

# Hardware Acceleration / Execution Providers (defaults to CPUExecutionProvider)
# Options: CPUExecutionProvider, OpenVINOExecutionProvider, CUDAExecutionProvider
PRIMARY_PROVIDER="CUDAExecutionProvider"
PRIMARY_PROVIDER_OPTIONS='{"device_id": 0}'
CPU_PROVIDER="CPUExecutionProvider"
CPU_FALLBACK_TTL=300
```

---

## 3. Running the Client

You can run the sync client using Docker. All operational files (the downloaded ONNX model, the chunk cache database, track checkpoints, and OpenVINO caches) live under the `data/` directory. You only need to mount this single directory to run the client.

Ensure your `./data` directory is prepared (it can be empty, in which case the container will download the model automatically on startup):
```bash
# Example local directory structure:
# ./data/
# └── (muq_v0.0.1.onnx will be downloaded here if not present)
```

### CPU-only Image
Run the sync client using the CPU-only image:
```bash
docker run --rm \
  --network="host" \
  --env-file .env \
  -v ./data:/app/data \
  ghcr.io/xoconoch/absmouth:latest
```

### CUDA Accelerated Image (NVIDIA GPU)
Build and run the sync client with NVIDIA GPU acceleration by passing NVIDIA devices
```bash
docker run --rm \
  --network="host" \
  --env-file .env \
  --device nvidia.com/gpu=all \
  -v ./data:/app/data \
  ghcr.io/xoconoch/absmouth:latest-cuda
```
Or build locally:
```bash
docker build -t absmouth:cuda -f Dockerfile.cuda .
```

### OpenVINO Accelerated Image (Intel GPU)
Run the sync client with Intel GPU hardware acceleration by sharing the `/dev/dri` device
```bash
docker run --rm \
  --network="host" \
  --env-file .env \
  --device /dev/dri \
  -v ./data:/app/data \
  ghcr.io/xoconoch/absmouth:latest-openvino
```

### Override Execution Settings
You can override configuration settings on-the-fly using environment variables via the `-e` flag in your `docker run` command:
* **Force Refetch Tracklist**: Force re-indexing metadata from Subsonic instead of using `checkpoint.json`.
  ```bash
  docker run --rm \
    --network="host" \
    --env-file .env \
    -e FORCE_REFETCH_TRACKLIST=true \
    -v ./data:/app/data \
    ghcr.io/xoconoch/absmouth:latest
  ```
* **Force Reprocess All**: Reprocess and re-generate embeddings for all tracks even if already synced.
  ```bash
  docker run --rm \
    --network="host" \
    --env-file .env \
    -e FORCE_REPROCESS_ALL=true \
    -v ./data:/app/data \
    ghcr.io/xoconoch/absmouth:latest
  ```
