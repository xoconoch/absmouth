# Absolute Pitch Subsonic Sync Client (absmouth)

`absmouth` is a lightweight, robust client that recursively crawls a Subsonic/OpenSubsonic server library, processes audio tracks in 10-second chunks, generates 1024-dimensional acoustic embeddings via ONNX Runtime, and synchronizes them with the Absolute Pitch vector database backend.

---

## 1. Prerequisites & Model Setup

This tool requires the `muq_large_dynamo.onnx` acoustic embedding model (~1.2 GB).

### Download the Model
Download the ONNX model from the following URL:
* **ONNX Model URL:** `[placeholder]`

Place the downloaded file (`muq_large_dynamo.onnx`) in your working directory, or define the environment variable `MODEL_PATH` pointing to the downloaded file.

---

## 2. Installation

You can run `absmouth` either in a development shell, via Nix, or by installing the package directly using standard Python tools.

### Option A: Nix Development Environment (Recommended)
If you are using Nix, you can enter the development shell which comes preconfigured with Python 3, ONNX Runtime (with OpenVINO acceleration support), `librosa`, and other required libraries:
```bash
nix develop
```

### Option B: Standard Python Package Installation
To install the package into your Python environment:
```bash
# Standard installation
pip install .

# Or for local development/editable mode:
pip install -e .
```

---

## 3. Configuration

Configure the application by creating a `.env` file in your working directory:

```env
# Subsonic Connection Parameters
SUBSONIC_URL="http://localhost:4533"
SUBSONIC_USER="your_subsonic_username"
SUBSONIC_PASS="your_subsonic_password"

# Absolute Pitch Target API Endpoint
API_URL="http://localhost:8100/tracks"

# ONNX Model Path (defaults to 'muq_large_dynamo.onnx' in the current working directory)
MODEL_PATH="muq_large_dynamo.onnx"

# Hardware Acceleration / Execution Providers
PRIMARY_PROVIDER="OpenVINOExecutionProvider"
CPU_PROVIDER="CPUExecutionProvider"
CPU_FALLBACK_TTL=300
```

---

## 4. Running the Client

Once installed or inside the Nix development shell, you can run the client in one of the following ways:

### Executing the CLI Command (if installed via pip)
```bash
absmouth
```

### Running as a Python Module
```bash
# Direct python execution
python -m absmouth.subsonic_client

# Inside the Nix shell without entering it first
nix develop --command python -m absmouth.subsonic_client
```

### Override Execution Settings
You can override configuration settings on-the-fly using environment variables:
* **Force Refetch Tracklist**: Force re-indexing metadata from Subsonic instead of using `checkpoint.json`.
  ```bash
  FORCE_REFETCH_TRACKLIST=true absmouth
  ```
* **Force Reprocess All**: Reprocess and re-generate embeddings for all tracks even if already synced.
  ```bash
  FORCE_REPROCESS_ALL=true absmouth
  ```
