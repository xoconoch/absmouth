import os
import json
import sys

def load_dotenv(dotenv_path=".env"):
    if os.path.exists(dotenv_path):
        with open(dotenv_path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line or line.startswith("#"):
                    continue
                if "=" in line:
                    key, val = line.split("=", 1)
                    key = key.strip()
                    val = val.strip()
                    if val.startswith(('"', "'")) and val.endswith(val[0]):
                        val = val[1:-1]
                    if key not in os.environ:
                        os.environ[key] = val

load_dotenv()

class Config:
    SUBSONIC_URL = os.environ.get("SUBSONIC_URL", "http://localhost:4040")
    SUBSONIC_USER = os.environ.get("SUBSONIC_USER", "")
    SUBSONIC_PASS = os.environ.get("SUBSONIC_PASS", "")
    SUBSONIC_API_VERSION = os.environ.get("SUBSONIC_API_VERSION", "1.16.1")
    SUBSONIC_CLIENT_NAME = os.environ.get("SUBSONIC_CLIENT_NAME", "AbsolutePitchClient")

    API_URL = os.environ.get("API_URL", "http://localhost:8100/tracks")
    API_REQUEST_TIMEOUT = int(os.environ.get("API_REQUEST_TIMEOUT", "30"))

    MODEL_PATH = os.environ.get("MODEL_PATH", "data/muq_v0.2.0.onnx")
    MODEL_DOWNLOAD_URL = os.environ.get("MODEL_DOWNLOAD_URL", "https://github.com/xoconoch/muq_onnx/releases/download/v0.2.0/muq_v0.2.0.onnx")
    # Hidden-state index to use. 0 is MuQ's conv feature output, which is the
    # best layer for centered cosine similarity; 12 is the final layer.
    MODEL_LAYER = int(os.environ.get("MODEL_LAYER", "0"))
    # Bump this to invalidate the chunk cache after changing pooling choices.
    CACHE_TAG = os.environ.get("CACHE_TAG", "")
    
    PRIMARY_PROVIDER = os.environ.get("PRIMARY_PROVIDER", "CPUExecutionProvider")
    PRIMARY_PROVIDER_OPTIONS_RAW = os.environ.get("PRIMARY_PROVIDER_OPTIONS", "")
    
    CPU_PROVIDER = os.environ.get("CPU_PROVIDER", "CPUExecutionProvider")
    CPU_PROVIDER_OPTIONS_RAW = os.environ.get("CPU_PROVIDER_OPTIONS", "")
    
    CPU_FALLBACK_TTL = int(os.environ.get("CPU_FALLBACK_TTL", "300"))

    CHUNK_CACHE_DB = os.environ.get("CHUNK_CACHE_DB", "data/chunk_cache.db")
    CHECKPOINT_FILE = os.environ.get("CHECKPOINT_FILE", "data/checkpoint.json")
    
    FORCE_REFETCH_TRACKLIST = os.environ.get("FORCE_REFETCH_TRACKLIST", "false").lower() in ("true", "1", "yes")
    FORCE_REPROCESS_ALL = os.environ.get("FORCE_REPROCESS_ALL", "false").lower() in ("true", "1", "yes")

    TARGET_SAMPLE_RATE = int(os.environ.get("TARGET_SAMPLE_RATE", "24000"))
    # MuQ v0.2.0 is exported with a static 30s input window.
    CHUNK_DURATION = int(os.environ.get("CHUNK_DURATION", "30"))
    # "uniform" or "energy" (weight chunks by their RMS).
    CHUNK_WEIGHTING = os.environ.get("CHUNK_WEIGHTING", "uniform").lower()
    # Chunks quieter than this RMS are ignored when other chunks exist.
    MIN_CHUNK_RMS = float(os.environ.get("MIN_CHUNK_RMS", "0.001"))

    @classmethod
    def get_primary_provider_options(cls):
        if cls.PRIMARY_PROVIDER_OPTIONS_RAW:
            try:
                return json.loads(cls.PRIMARY_PROVIDER_OPTIONS_RAW)
            except Exception as e:
                print(f"Error parsing PRIMARY_PROVIDER_OPTIONS JSON: {e}")
        
        if cls.PRIMARY_PROVIDER == "OpenVINOExecutionProvider":
            return {
                "device_type": "GPU",
                "precision": "FP32",
                "cache_dir": "data/ov_cache"
            }
        elif cls.PRIMARY_PROVIDER == "CUDAExecutionProvider":
            return {
                "device_id": 0,
            }
        return {}

    @classmethod
    def get_cpu_provider_options(cls):
        if cls.CPU_PROVIDER_OPTIONS_RAW:
            try:
                return json.loads(cls.CPU_PROVIDER_OPTIONS_RAW)
            except Exception as e:
                print(f"Error parsing CPU_PROVIDER_OPTIONS JSON: {e}")
        
        if cls.CPU_PROVIDER == "OpenVINOExecutionProvider":
            return {"device_type": "CPU"}
        return {}

    @classmethod
    def validate(cls):
        missing = []
        if not cls.SUBSONIC_USER:
            missing.append("SUBSONIC_USER")
        if not cls.SUBSONIC_PASS:
            missing.append("SUBSONIC_PASS")
        if missing:
            print(f"Error: Missing required environment variables: {', '.join(missing)}")
            print("Please set them in your environment or write them to a .env file.")
            sys.exit(1)
        if cls.CHUNK_DURATION < 1:
            print("Error: CHUNK_DURATION must be at least 1 second.")
            sys.exit(1)
        if cls.MODEL_LAYER < 0:
            print("Error: MODEL_LAYER must be zero or greater.")
            sys.exit(1)
        if cls.CHUNK_WEIGHTING not in ("uniform", "energy"):
            print("Error: CHUNK_WEIGHTING must be 'uniform' or 'energy'.")
            sys.exit(1)
        if cls.MIN_CHUNK_RMS < 0:
            print("Error: MIN_CHUNK_RMS must be zero or greater.")
            sys.exit(1)
