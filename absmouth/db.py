import os
import json
import sqlite3
import hashlib

def get_stable_id(text: str) -> str:
    """Generates a consistent, reproducible string ID from text metadata."""
    if not text:
        return "unknown"
    return hashlib.md5(text.strip().lower().encode('utf-8')).hexdigest()

class CheckpointManager:
    """Manages tracking/resuming state to survive restarts and failures."""
    def __init__(self, filepath):
        self.filepath = filepath
        self.data = {
            "tracks": [],
            "completed_ids": []
        }
        self.load()

    def load(self):
        if os.path.exists(self.filepath):
            try:
                with open(self.filepath, "r", encoding="utf-8") as f:
                    self.data = json.load(f)
                if "tracks" not in self.data:
                    self.data["tracks"] = []
                if "completed_ids" not in self.data:
                    self.data["completed_ids"] = []
                print(f"Checkpoint loaded. Tracks: {len(self.data['tracks'])}, Completed: {len(self.data['completed_ids'])}")
            except Exception as e:
                print(f"Warning: Failed to load checkpoint {self.filepath}: {e}. Starting fresh.")
                self.data = {"tracks": [], "completed_ids": []}

    def save(self):
        try:
            with open(self.filepath, "w", encoding="utf-8") as f:
                json.dump(self.data, f, indent=2, ensure_ascii=False)
        except Exception as e:
            print(f"Error saving checkpoint to {self.filepath}: {e}")

    def save_tracklist(self, tracks):
        self.data["tracks"] = tracks
        self.save()

    def mark_completed(self, track_id):
        if track_id not in self.data["completed_ids"]:
            self.data["completed_ids"].append(track_id)
            self.save()

class ChunkCache:
    """Persistent database mapping chunk hashes to precomputed float embeddings."""
    def __init__(self, db_path):
        self.db_path = db_path
        self._init_db()

    def _init_db(self):
        conn = sqlite3.connect(self.db_path)
        try:
            conn.execute("""
                CREATE TABLE IF NOT EXISTS chunk_cache (
                    chunk_hash TEXT PRIMARY KEY,
                    embedding TEXT
                )
            """)
            conn.commit()
        finally:
            conn.close()

    def get(self, chunk_hash):
        conn = sqlite3.connect(self.db_path)
        try:
            cursor = conn.cursor()
            cursor.execute("SELECT embedding FROM chunk_cache WHERE chunk_hash = ?", (chunk_hash,))
            row = cursor.fetchone()
            if row:
                try:
                    return json.loads(row[0])
                except Exception as e:
                    print(f"Warning: Failed to decode cached chunk embedding: {e}")
                    return None
        finally:
            conn.close()
        return None

    def set(self, chunk_hash, embedding_list):
        try:
            val = json.dumps(embedding_list)
            conn = sqlite3.connect(self.db_path)
            try:
                conn.execute(
                    "INSERT OR REPLACE INTO chunk_cache (chunk_hash, embedding) VALUES (?, ?)",
                    (chunk_hash, val)
                )
                conn.commit()
            finally:
                conn.close()
        except Exception as e:
            print(f"Warning: Failed to cache chunk embedding to database: {e}")
