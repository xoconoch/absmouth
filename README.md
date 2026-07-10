# Absolute Pitch Subsonic Sync Client

A lightweight, robust client that recursively crawls a Subsonic/OpenSubsonic server library, processes audio tracks in 10-second chunks, generates 1024-dimensional acoustic embeddings via ONNX Runtime, and synchronizes them with the Absolute Pitch vector database backend.

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
python subsonic_client.py
```

### 3. Execution Options
You can override settings via temporary environment variables:
- **Force Refetch Tracklist**: `FORCE_REFETCH_TRACKLIST=true nix develop --command python subsonic_client.py`
- **Force Reprocess All**: `FORCE_REPROCESS_ALL=true nix develop --command python subsonic_client.py`
