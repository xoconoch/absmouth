import os
import time
import hashlib
import shutil
import subprocess
import numpy as np
import soundfile as sf
import soxr
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

    @staticmethod
    def _decode_with_ffmpeg(file_path, target_sr):
        """Fallback decoder for formats libsndfile cannot read (e.g. AAC/M4A)."""
        ffmpeg = shutil.which("ffmpeg")
        if not ffmpeg:
            raise RuntimeError(
                f"Unsupported audio format for {file_path!r} and ffmpeg is not installed."
            )
        proc = subprocess.run(
            [
                ffmpeg, "-v", "error", "-i", file_path,
                "-f", "f32le", "-ac", "1", "-ar", str(target_sr), "-",
            ],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
        )
        if proc.returncode != 0:
            raise RuntimeError(
                f"ffmpeg failed to decode {file_path!r}: "
                f"{proc.stderr.decode('utf-8', 'replace').strip()}"
            )
        wav = np.frombuffer(proc.stdout, dtype=np.float32).reshape(-1, 1)
        return wav, target_sr

    def load_and_preprocess_audio(self, file_path):
        """Loads and converts the file to the target samplerate and tensor format."""
        target_sr = self.config.TARGET_SAMPLE_RATE
        try:
            wav, sr = sf.read(file_path, dtype="float32", always_2d=True)
        except Exception:
            wav, sr = self._decode_with_ffmpeg(file_path, target_sr)

        if wav.shape[1] > 1:
            wav = wav.mean(axis=1)
        else:
            wav = wav[:, 0]

        if sr != target_sr:
            wav = soxr.resample(wav, sr, target_sr, quality="HQ")

        return np.expand_dims(np.asarray(wav, dtype=np.float32), axis=0)

    @staticmethod
    def _normalize(vec):
        norm = float(np.linalg.norm(vec))
        if norm > 0:
            return vec / norm
        return vec

    def _pool_chunk(self, session, input_name, output_name, chunk, chunk_hash):
        """Return the time-pooled embedding for one chunk, using the cache."""
        cached_emb = self.chunk_cache.get(chunk_hash)
        if cached_emb is not None:
            cached_emb_arr = np.array(cached_emb, dtype=np.float32)
            if not np.isnan(cached_emb_arr).any():
                return cached_emb_arr
            print("  [Warning] Cached chunk embedding had NaNs; recomputing.")

        outputs = session.run([output_name], {input_name: chunk})
        last_hidden_state = outputs[0]

        if np.isnan(last_hidden_state).any():
            print("  [Warning] NaN found in raw inference outputs.")
            return None

        chunk_emb = np.mean(last_hidden_state, axis=1).flatten()

        if np.isnan(chunk_emb).any():
            print("  [Warning] NaN found in pooled chunk embedding.")
            return None

        self.chunk_cache.set(chunk_hash, chunk_emb.tolist())
        return chunk_emb

    def run_inference_on_session(self, session, audio_data):
        """
        Processes audio in chunks, normalizes each chunk embedding, and pools
        them. Returns None if a NaN value is encountered at any stage.
        """
        input_meta = session.get_inputs()[0]
        input_name = input_meta.name
        chunk_size = self.config.TARGET_SAMPLE_RATE * self.config.CHUNK_DURATION

        # Fail fast when CHUNK_DURATION does not match the static ONNX input.
        if len(input_meta.shape) == 2 and isinstance(input_meta.shape[1], int):
            expected = input_meta.shape[1]
            if expected != chunk_size:
                raise ValueError(
                    f"Model expects {expected} samples per window, but CHUNK_DURATION="
                    f"{self.config.CHUNK_DURATION}s at {self.config.TARGET_SAMPLE_RATE} Hz "
                    f"gives {chunk_size}. Set CHUNK_DURATION to match the model export."
                )

        output_names = [o.name for o in session.get_outputs()]
        desired_output = f"layer_{self.config.MODEL_LAYER}"
        if desired_output in output_names:
            output_name = desired_output
        else:
            output_name = output_names[min(self.config.MODEL_LAYER, len(output_names) - 1)]
            print(
                f"  [Warning] Model has no output {desired_output}; "
                f"using {output_name}."
            )

        cache_signature = (
            f"{self.config.MODEL_PATH}|layer={self.config.MODEL_LAYER}|"
            f"sr={self.config.TARGET_SAMPLE_RATE}|chunk={self.config.CHUNK_DURATION}|"
            f"tag={self.config.CACHE_TAG}|"
        ).encode("utf-8")

        total_samples = audio_data.shape[1]
        weighted_sum = None
        weight_sum = 0.0
        total_chunks = 0
        skipped_silent = 0
        loudest_chunk = None
        loudest_chunk_hash = None
        loudest_rms = -1.0

        for i in range(0, total_samples, chunk_size):
            total_chunks += 1
            chunk = audio_data[:, i:i+chunk_size]
            rms = float(np.sqrt(np.mean(np.square(chunk), dtype=np.float64)))

            if chunk.shape[1] < chunk_size:
                padding_needed = chunk_size - chunk.shape[1]
                chunk = np.pad(chunk, ((0, 0), (0, padding_needed)), mode='constant')

            chunk_hash = hashlib.sha256(cache_signature + chunk.tobytes()).hexdigest()
            if rms > loudest_rms:
                loudest_rms = rms
                loudest_chunk = chunk
                loudest_chunk_hash = chunk_hash

            if rms < self.config.MIN_CHUNK_RMS:
                skipped_silent += 1
                continue

            chunk_emb = self._pool_chunk(
                session, input_name, output_name, chunk, chunk_hash
            )
            if chunk_emb is None:
                return None

            # Normalizing each chunk stops loud sections from dominating the
            # track centroid while still letting energy weighting bias the mix.
            chunk_emb = self._normalize(chunk_emb)
            weight = rms if self.config.CHUNK_WEIGHTING == "energy" else 1.0
            if weighted_sum is None:
                weighted_sum = np.zeros_like(chunk_emb, dtype=np.float64)
            weighted_sum += chunk_emb.astype(np.float64) * weight
            weight_sum += weight

        if weighted_sum is None:
            # The whole track is below the silence threshold. Fall back to the
            # loudest chunk so quiet recordings still get an embedding.
            if loudest_chunk is None:
                return None
            print("  [Warning] Every chunk was below MIN_CHUNK_RMS; using the loudest chunk.")
            chunk_emb = self._pool_chunk(
                session, input_name, output_name, loudest_chunk, loudest_chunk_hash
            )
            if chunk_emb is None:
                return None
            weighted_sum = self._normalize(chunk_emb).astype(np.float64)
            weight_sum = 1.0
        elif skipped_silent:
            print(f"  [Info] Skipped {skipped_silent}/{total_chunks} silent chunks.")

        mean_embedding = (weighted_sum / weight_sum).astype(np.float32)
        mean_embedding = self._normalize(mean_embedding).astype(np.float32)

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
