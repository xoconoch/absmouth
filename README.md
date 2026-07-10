# Absolute Pitch Subsonic Sync Client

A lightweight, robust client that recursively crawls a Subsonic/OpenSubsonic server library, processes audio tracks in 10-second chunks, generates 1024-dimensional acoustic embeddings via ONNX Runtime, and synchronizes them with the Absolute Pitch vector database backend.

## Project Structure

- `subsonic_client.py`: Main entrypoint coordinating the crawl-and-embed pipeline.
- `config.py`: Loads environment configurations (from env/`.env`) and validates variables.
- `db.py`: Implements checkpointing (`checkpoint.json`) and the 10s chunk embedding cache (`chunk_cache.db`).
- `subsonic.py`: Queries the Subsonic API to crawl folders and stream audio files.
- `abspitch.py`: Sends completed track metadata and embeddings to the Absolute Pitch backend.
- `model.py`: Preprocesses audio and runs ONNX inference with lazy, per-track CPU fallback (with TTL resource releasing).

## Quick Start

### 1. Configuration
Create a `.env` file in this directory:
```bash
SUBSONIC_URL="http://localhost:4040"
SUBSONIC_USER="your_subsonic_username"
SUBSONIC_PASS="your_subsonic_password"
API_URL="http://localhost:8000/tracks"
```

### 2. Run Sync
Execute the client using the pre-configured Nix environment:
```bash
nix develop --command python subsonic_client.py
```

### 3. Execution Options
You can override settings via temporary environment variables:
- **Force Refetch Tracklist**: `FORCE_REFETCH_TRACKLIST=true nix develop --command python subsonic_client.py`
- **Force Reprocess All**: `FORCE_REPROCESS_ALL=true nix develop --command python subsonic_client.py`
