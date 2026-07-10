import os
import json
import sys

# Helper to load .env file if it exists in the workspace
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
                    # Remove surrounding quotes if present
                    if val.startswith(('"', "'")) and val.endswith(val[0]):
                        val = val[1:-1]
                    if key not in os.environ:
                        os.environ[key] = val

load_dotenv()

class Config:
    # Subsonic connection parameters
    SUBSONIC_URL = os.environ.get("SUBSONIC_URL", "http://localhost:4040")
    SUBSONIC_USER = os.environ.get("SUBSONIC_USER", "")
    SUBSONIC_PASS = os.environ.get("SUBSONIC_PASS", "")
    SUBSONIC_API_VERSION = os.environ.get("SUBSONIC_API_VERSION", "1.16.1")
    SUBSONIC_CLIENT_NAME = os.environ.get("SUBSONIC_CLIENT_NAME", "AbsolutePitchClient")

    # Absolute Pitch target API
    API_URL = os.environ.get("API_URL", "http://localhost:8000/tracks")

    # ONNX Model parameters
    MODEL_PATH = os.environ.get("MODEL_PATH", "muq_large_dynamo.onnx")
    
    # Primary Execution Provider (Hardware Acceleration)
    PRIMARY_PROVIDER = os.environ.get("PRIMARY_PROVIDER", "OpenVINOExecutionProvider")
    PRIMARY_PROVIDER_OPTIONS_RAW = os.environ.get("PRIMARY_PROVIDER_OPTIONS", "")
    
    # Fallback Execution Provider (usually CPU)
    CPU_PROVIDER = os.environ.get("CPU_PROVIDER", "CPUExecutionProvider")
    CPU_PROVIDER_OPTIONS_RAW = os.environ.get("CPU_PROVIDER_OPTIONS", "")
    
    # Time-to-live for CPU session in seconds.
    CPU_FALLBACK_TTL = int(os.environ.get("CPU_FALLBACK_TTL", "300"))

    # Local storage configurations
    CHUNK_CACHE_DB = os.environ.get("CHUNK_CACHE_DB", "chunk_cache.db")
    CHECKPOINT_FILE = os.environ.get("CHECKPOINT_FILE", "checkpoint.json")
    
    # Processing controls
    FORCE_REFETCH_TRACKLIST = os.environ.get("FORCE_REFETCH_TRACKLIST", "false").lower() in ("true", "1", "yes")
    FORCE_REPROCESS_ALL = os.environ.get("FORCE_REPROCESS_ALL", "false").lower() in ("true", "1", "yes")

    # Audio preprocessing parameters
    TARGET_SAMPLE_RATE = int(os.environ.get("TARGET_SAMPLE_RATE", "24000"))
    CHUNK_DURATION = int(os.environ.get("CHUNK_DURATION", "10"))

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
                "cache_dir": "./ov_cache"
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
