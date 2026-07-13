import time
import hashlib
import numpy as np
import librosa
import onnxruntime as ort

class ModelSessionManager:
    """Manages execution provider sessions including lazy CPU loading with TTL."""
    def __init__(self, config):
        self.config = config
        self.model_path = config.MODEL_PATH
        self.primary_session = None
        self.cpu_session = None
        self.cpu_last_used = 0.0
        self._ensure_model_exists()

    def _ensure_model_exists(self):
        import os
        import urllib.request

        if os.path.exists(self.model_path):
            return

        url = self.config.MODEL_DOWNLOAD_URL
        print(f"Model not found at {self.model_path}. Automatically downloading from {url}...")
        
        dir_name = os.path.dirname(self.model_path)
        if dir_name:
            os.makedirs(dir_name, exist_ok=True)
            
        temp_target_path = self.model_path + ".tmp"
        try:
            req = urllib.request.Request(
                url, 
                headers={'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64)'}
            )
            with urllib.request.urlopen(req) as response, open(temp_target_path, "wb") as out_file:
                meta = response.info()
                content_length = meta.get("Content-Length")
                file_size = int(content_length) if content_length else None
                
                downloaded = 0
                block_size = 65536
                last_percentage = -1
                
                while True:
                    buffer = response.read(block_size)
                    if not buffer:
                        break
                    downloaded += len(buffer)
                    out_file.write(buffer)
                    
                    if file_size:
                        percentage = int(downloaded * 100 / file_size)
                        if percentage != last_percentage:
                            if percentage % 5 == 0:
                                print(f"Downloading: {percentage}% ({downloaded / (1024*1024):.1f} MB / {file_size / (1024*1024):.1f} MB)", end="\r", flush=True)
                                last_percentage = percentage
                    else:
                        if downloaded % (1024 * 1024 * 5) == 0:
                            print(f"Downloaded {downloaded / (1024*1024):.1f} MB...", end="\r", flush=True)
                print()  
                
            os.replace(temp_target_path, self.model_path)
            print(f"Model downloaded successfully and saved to {self.model_path}")
        except Exception as e:
            if os.path.exists(temp_target_path):
                try:
                    os.remove(temp_target_path)
                except Exception:
                    pass
            print(f"CRITICAL: Failed to download model: {e}")
            raise e

    def get_primary_session(self):
        if self.primary_session is None:
            provider = self.config.PRIMARY_PROVIDER
            options = self.config.get_primary_provider_options()
            
            print(f"Initializing primary ONNX session ({provider})...")
            try:
                self.primary_session = ort.InferenceSession(
                    self.model_path,
                    providers=[provider],
                    provider_options=[options] if options else None
                )
                print("Primary ONNX session loaded successfully.")
            except Exception as e:
                print(f"Failed to load primary session with {provider}: {e}")
                print("Falling back to CPU fallback list...")
                raise e
        return self.primary_session

    def get_cpu_session(self):
        self.cpu_last_used = time.time()
        if self.cpu_session is None:
            provider = self.config.CPU_PROVIDER
            options = self.config.get_cpu_provider_options()
            
            print(f"Lazy-initializing fallback CPU ONNX session ({provider})...")
            try:
                self.cpu_session = ort.InferenceSession(
                    self.model_path,
                    providers=[provider],
                    provider_options=[options] if options else None
                )
                print("Fallback CPU session loaded successfully.")
            except Exception as e:
                print(f"Failed to initialize CPU session: {e}")
                raise e
        return self.cpu_session

    def check_cpu_ttl(self):
        """Unloads CPU model if TTL expired, to conserve memory."""
        if self.cpu_session is not None and self.config.CPU_FALLBACK_TTL > 0:
            elapsed = time.time() - self.cpu_last_used
            if elapsed > self.config.CPU_FALLBACK_TTL:
                print(f"CPU fallback session has been idle for {elapsed:.1f}s (TTL: {self.config.CPU_FALLBACK_TTL}s). Unloading to free memory.")
                self.cpu_session = None
                import gc
                gc.collect()

    def close_all(self):
        self.primary_session = None
        self.cpu_session = None
        import gc
        gc.collect()

class EmbeddingEngine:
    """Manages audio loading, caching, running inference, and fallback safety."""
    def __init__(self, config, model_manager, chunk_cache):
        self.config = config
        self.model_manager = model_manager
        self.chunk_cache = chunk_cache

    def load_and_preprocess_audio(self, file_path):
        """Loads and converts the file to the target samplerate and tensor format."""
        wav, sr = librosa.load(file_path, sr=self.config.TARGET_SAMPLE_RATE)
        wav = np.expand_dims(wav, axis=0).astype(np.float32)
        return wav

    def run_inference_on_session(self, session, audio_data):
        """
        Processes audio in chunks. Checkpoints and pools embeddings.
        Returns None if a NaN value is encountered at any stage.
        """
        input_name = session.get_inputs()[0].name
        chunk_size = self.config.TARGET_SAMPLE_RATE * self.config.CHUNK_DURATION
        total_samples = audio_data.shape[1]
        chunk_embeddings = []
        
        for i in range(0, total_samples, chunk_size):
            chunk = audio_data[:, i:i+chunk_size]
            
            if chunk.shape[1] < chunk_size:
                padding_needed = chunk_size - chunk.shape[1]
                chunk = np.pad(chunk, ((0, 0), (0, padding_needed)), mode='constant')
            
            chunk_bytes = chunk.tobytes()
            chunk_hash = hashlib.sha256(chunk_bytes).hexdigest()
            
            cached_emb = self.chunk_cache.get(chunk_hash)
            if cached_emb is not None:
                cached_emb_arr = np.array(cached_emb, dtype=np.float32)
                if not np.isnan(cached_emb_arr).any():
                    chunk_embeddings.append(cached_emb_arr)
                    continue
                else:
                    print("  [Warning] Cached chunk embedding had NaNs; recomputing.")

            outputs = session.run(None, {input_name: chunk})
            last_hidden_state = outputs[0]
            
            if np.isnan(last_hidden_state).any():
                print("  [Warning] NaN found in raw inference outputs.")
                return None
                
            chunk_emb = np.mean(last_hidden_state, axis=1).flatten()
            
            if np.isnan(chunk_emb).any():
                print("  [Warning] NaN found in pooled chunk embedding.")
                return None

            self.chunk_cache.set(chunk_hash, chunk_emb.tolist())
            chunk_embeddings.append(chunk_emb)
            
        if not chunk_embeddings:
            return None
            
        mean_embedding = np.mean(chunk_embeddings, axis=0)
        
        if np.isnan(mean_embedding).any():
            print("  [Warning] NaN found in final aggregated embedding.")
            return None
            
        return mean_embedding

    def compute_embedding(self, audio_data):
        """Attempts fast hardware acceleration first, falls back to CPU if NaNs occur."""
        primary_session = self.model_manager.get_primary_session()
        embedding = self.run_inference_on_session(primary_session, audio_data)
        
        if embedding is None:
            print("  [Warning] NaN detected during primary inference. Falling back to CPU...")
            cpu_session = self.model_manager.get_cpu_session()
            embedding = self.run_inference_on_session(cpu_session, audio_data)
            
            if embedding is None or np.isnan(embedding).any():
                raise ValueError("Acoustic model produced NaNs on both hardware accelerator and CPU fallback.")
                
        return embedding
