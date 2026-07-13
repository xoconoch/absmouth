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

    for path in [Config.CHUNK_CACHE_DB, Config.CHECKPOINT_FILE]:
        dir_name = os.path.dirname(path)
        if dir_name:
            os.makedirs(dir_name, exist_ok=True)

    temp_dir = os.path.join(os.getcwd(), "tmp_audio")
    os.makedirs(temp_dir, exist_ok=True)
    print(f"Temporary audio directory: {temp_dir}")

    subsonic = SubsonicClient(Config)
    abspitch = AbsolutePitchClient(Config)
    checkpoint = CheckpointManager(Config.CHECKPOINT_FILE)
    chunk_cache = ChunkCache(Config.CHUNK_CACHE_DB)
    model_manager = ModelSessionManager(Config)
    engine = EmbeddingEngine(Config, model_manager, chunk_cache)

    print("Fetching list of artists from Subsonic...")
    try:
        artists = subsonic.get_artists()
    except Exception as e:
        print(f"CRITICAL: Failed to retrieve artists from Subsonic: {e}")
        sys.exit(1)

    total_artists = len(artists)
    print(f"Found {total_artists} artists to process.")

    completed_ids = set(checkpoint.data["completed_ids"])
    if Config.FORCE_REPROCESS_ALL:
        print("FORCE_REPROCESS_ALL is active. Reprocessing all files.")
        completed_ids.clear()

    print(f"Already completed tracks: {len(completed_ids)}")

    success_count = 0
    failure_count = 0

    try:
        for a_idx, artist in enumerate(artists, 1):
            artist_id = artist.get("id")
            artist_name = artist.get("name", "Unknown Artist")
            if not artist_id:
                continue

            print(f"\n[{a_idx}/{total_artists}] Crawling artist: {artist_name}")
            
            try:
                albums = subsonic.get_artist(artist_id)
            except Exception as e:
                print(f"  Warning: Failed to fetch albums for artist {artist_name}: {e}")
                continue

            artist_songs = []
            for album in albums:
                album_id = album.get("id")
                album_name = album.get("name", "Unknown Album")
                if not album_id:
                    continue
                try:
                    songs = subsonic.get_album(album_id)
                    for song in songs:
                        if "artist" not in song and artist_name:
                            song["artist"] = artist_name
                        if "artistId" not in song and artist_id:
                            song["artistId"] = artist_id
                        if "album" not in song and album_name:
                            song["album"] = album_name
                        artist_songs.append(song)
                except Exception as e:
                    print(f"  Warning: Failed to crawl album {album_name} ({album_id}): {e}")

            songs_to_process = [s for s in artist_songs if str(s.get("id")) not in completed_ids]
            
            if not songs_to_process:
                continue

            print(f"  Found {len(songs_to_process)} tracks to process for artist '{artist_name}' (Total in library for this artist: {len(artist_songs)})")

            for s_idx, track in enumerate(songs_to_process, 1):
                track_id = str(track.get("id"))
                title = track.get("title") or "Unknown Title"
                album_name = track.get("album") or "Unknown Album"
                
                print(f"  ({s_idx}/{len(songs_to_process)}) Processing track '{title}' (ID: {track_id})")
                
                temp_file_path = os.path.join(temp_dir, f"track_{track_id}.tmp")
                
                try:
                    print("    Downloading audio stream...")
                    subsonic.download_track(track_id, temp_file_path)
                    
                    print("    Preprocessing audio...")
                    audio_data = engine.load_and_preprocess_audio(temp_file_path)
                    
                    print("    Running ONNX model inference...")
                    embedding = engine.compute_embedding(audio_data)
                    
                    print("    Pushing to Absolute Pitch API...")
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
                        completed_ids.add(track_id)
                        success_count += 1
                        print(f"    [Success] Track '{title}' ingested successfully.")
                    else:
                        failure_count += 1
                        print(f"    [Error {response.status_code}] Failed to ingest: {response.text}")

                except KeyboardInterrupt:
                    print("\nSync execution interrupted by user. Saving states and clean-exiting...")
                    raise
                except Exception as e:
                    failure_count += 1
                    print(f"    [Failed] Error processing file: {e}")
                finally:
                    if os.path.exists(temp_file_path):
                        try:
                            os.remove(temp_file_path)
                        except Exception as e:
                            print(f"    [Warning] Clean-up failed for {temp_file_path}: {e}")
                    
                    model_manager.check_cpu_ttl()

    except KeyboardInterrupt:
        pass
    finally:
        if os.path.exists(temp_dir) and not os.listdir(temp_dir):
            try:
                os.rmdir(temp_dir)
            except Exception:
                pass
        
        model_manager.close_all()
        print(f"\nSync complete. Successes: {success_count}, Failures: {failure_count}.")

if __name__ == "__main__":
    main()
