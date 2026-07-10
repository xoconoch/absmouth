#!/usr/bin/env python3
"""
Absolute Pitch Subsonic Sync Client
Entrypoint script that coordinates the synchronization loop.
"""

import os
import sys

from .config import Config
from .db import CheckpointManager, ChunkCache, get_stable_id
from .subsonic import SubsonicClient
from .abspitch import AbsolutePitchClient
from .model import ModelSessionManager, EmbeddingEngine

def main():
    Config.validate()
    print("--- Absolute Pitch Subsonic Sync Client ---")
    print(f"Connecting to Subsonic: {Config.SUBSONIC_URL}")
    print(f"API Target: {Config.API_URL}")
    print(f"Model Path: {Config.MODEL_PATH}")
    print(f"Primary Acceleration: {Config.PRIMARY_PROVIDER}")
    print(f"CPU Fallback: {Config.CPU_PROVIDER} (TTL: {Config.CPU_FALLBACK_TTL}s)")
    print(f"Chunk Cache: {Config.CHUNK_CACHE_DB}")
    print(f"Checkpoint File: {Config.CHECKPOINT_FILE}")

    # Ensure parent directories exist for cache and checkpoint storage
    for path in [Config.CHUNK_CACHE_DB, Config.CHECKPOINT_FILE]:
        dir_name = os.path.dirname(path)
        if dir_name:
            os.makedirs(dir_name, exist_ok=True)

    # Set up workspace-confined temporary directory for downloaded audio streams
    temp_dir = os.path.join(os.getcwd(), "tmp_audio")
    os.makedirs(temp_dir, exist_ok=True)
    print(f"Temporary audio directory: {temp_dir}")

    # Initialize components
    subsonic = SubsonicClient(Config)
    abspitch = AbsolutePitchClient(Config)
    checkpoint = CheckpointManager(Config.CHECKPOINT_FILE)
    chunk_cache = ChunkCache(Config.CHUNK_CACHE_DB)
    model_manager = ModelSessionManager(Config)
    engine = EmbeddingEngine(Config, model_manager, chunk_cache)

    # Resolve tracklist
    if Config.FORCE_REFETCH_TRACKLIST or not checkpoint.data["tracks"]:
        print("Fetching full track list from Subsonic...")
        try:
            tracks = subsonic.crawl_all_tracks()
            checkpoint.save_tracklist(tracks)
        except Exception as e:
            print(f"CRITICAL: Failed to retrieve tracks from Subsonic: {e}")
            sys.exit(1)
    else:
        print("Using cached track list from checkpoint file.")
        
    tracks_to_process = checkpoint.data["tracks"]
    completed_ids = set(checkpoint.data["completed_ids"])

    if Config.FORCE_REPROCESS_ALL:
        print("FORCE_REPROCESS_ALL is active. Reprocessing all files.")
        completed_ids.clear()

    # Filter out already completed items
    filtered_tracks = [t for t in tracks_to_process if str(t.get("id")) not in completed_ids]
    total_to_process = len(filtered_tracks)
    print(f"Total tracks in list: {len(tracks_to_process)}")
    print(f"Already completed: {len(completed_ids)}")
    print(f"Remaining to process: {total_to_process}")

    if total_to_process == 0:
        print("All tracks successfully synchronized. Exiting.")
        return

    success_count = 0
    failure_count = 0

    try:
        for idx, track in enumerate(filtered_tracks, 1):
            track_id = str(track.get("id"))
            title = track.get("title") or "Unknown Title"
            artist_name = track.get("artist") or "Unknown Artist"
            album_name = track.get("album") or "Unknown Album"
            
            # Establish standard subsonic client metadata
            artist_id = track.get("artistId")
            if not artist_id:
                artist_id = get_stable_id(artist_name)
            else:
                artist_id = str(artist_id)

            print(f"\n[{idx}/{total_to_process}] Processing track '{title}' by '{artist_name}' (ID: {track_id})")
            
            temp_file_path = os.path.join(temp_dir, f"track_{track_id}.tmp")
            
            try:
                # 1. Download file
                print("  Downloading audio stream...")
                subsonic.download_track(track_id, temp_file_path)
                
                # 2. Preprocess audio
                print("  Preprocessing audio...")
                audio_data = engine.load_and_preprocess_audio(temp_file_path)
                
                # 3. Generate embedding (with automatic lazy fallback to CPU)
                print("  Running ONNX model inference...")
                embedding = engine.compute_embedding(audio_data)
                
                # 4. Push to REST API
                print("  Pushing to Absolute Pitch API...")
                response = abspitch.insert_track(
                    subsonic_id=track_id,
                    title=title,
                    album_name=album_name,
                    artist_id=artist_id,
                    artist_name=artist_name,
                    embedding=embedding.tolist()
                )
                
                if response.status_code in (200, 201):
                    checkpoint.mark_completed(track_id)
                    success_count += 1
                    print(f"  [Success] Track '{title}' ingested successfully.")
                else:
                    failure_count += 1
                    print(f"  [Error {response.status_code}] Failed to ingest: {response.text}")

            except KeyboardInterrupt:
                print("\nSync execution interrupted by user. Saving states and clean-exiting...")
                raise
            except Exception as e:
                failure_count += 1
                print(f"  [Failed] Error processing file: {e}")
            finally:
                # Cleanup audio files to ensure workspace directories don't bloat
                if os.path.exists(temp_file_path):
                    try:
                        os.remove(temp_file_path)
                    except Exception as e:
                        print(f"  [Warning] Clean-up failed for {temp_file_path}: {e}")
                
                # Check CPU model session TTL to free memory
                model_manager.check_cpu_ttl()

    except KeyboardInterrupt:
        pass
    finally:
        # Cleanup temporary files folder
        if os.path.exists(temp_dir) and not os.listdir(temp_dir):
            try:
                os.rmdir(temp_dir)
            except Exception:
                pass
        
        # Release model resources
        model_manager.close_all()
        print(f"\nSync complete. Successes: {success_count}, Failures: {failure_count}.")

if __name__ == "__main__":
    main()
