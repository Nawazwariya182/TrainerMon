import os
import secrets
from pathlib import Path

# Paths
BASE_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = BASE_DIR.parent
MODELS_DIR = PROJECT_ROOT / "MODELS"

# Generate or load secure API Key
ENV_FILE = BASE_DIR / ".env"
DEFAULT_API_KEY = "sk-mltrain-" + secrets.token_hex(16)

def get_or_create_api_key() -> str:
    if ENV_FILE.exists():
        with open(ENV_FILE, "r", encoding="utf-8") as f:
            for line in f:
                if line.startswith("ML_TRAIN_API_KEY="):
                    key = line.strip().split("=", 1)[1].strip()
                    if key:
                        return key
    
    # Save newly generated key
    with open(ENV_FILE, "w", encoding="utf-8") as f:
        f.write(f"# ML Train Studio API Key\nML_TRAIN_API_KEY={DEFAULT_API_KEY}\n")
    return DEFAULT_API_KEY

API_KEY = get_or_create_api_key()

# Model Registry
AVAILABLE_MODELS = {
    "qwen3-8b": {
        "id": "qwen3-8b",
        "name": "Qwen3-8B-Q8_0",
        "path": MODELS_DIR / "Qwen3-8B-Q8_0.gguf",
        "mmproj": None,
        "default_ctx": 16384,
        "max_ctx": 32768,
        "max_tokens_cap": 8192,
        "description": "High-precision Q8_0 quantized Qwen3 8B parameter model",
    },
    "qwen3.5-9b": {
        "id": "qwen3.5-9b",
        "name": "Qwen3.5-9B-Q8_0",
        "path": MODELS_DIR / "Qwen3.5-9B-Q8_0.gguf",
        "mmproj": None,
        "default_ctx": 16384,
        "max_ctx": 32768,
        "max_tokens_cap": 8192,
        "description": "Next-generation Qwen3.5 9B parameter model with extended context",
    },
    "qwen3-vl-8b": {
        "id": "qwen3-vl-8b",
        "name": "Qwen3-VL-8B-Instruct-Q8_0",
        "path": MODELS_DIR / "Qwen3-VL-8B-Instruct-GGUF" / "Qwen3-VL-8B-Instruct-Q8_0.gguf",
        "mmproj": MODELS_DIR / "Qwen3-VL-8B-Instruct-GGUF" / "mmproj-Qwen3-VL-8B-Instruct-F16.gguf",
        "default_ctx": 16384,
        "max_ctx": 32768,
        "max_tokens_cap": 8192,
        "description": "Vision-Language multimodal model with F16 multimodal projector",
    },
}

# Default Server Host & Port
HOST = os.getenv("HOST", "0.0.0.0")
PORT = int(os.getenv("PORT", 8000))
DEFAULT_MODEL_ID = "qwen3-8b"

# Auto-offload idle timeout in seconds (Default: 120s / 2 minutes)
IDLE_TIMEOUT_SECONDS = int(os.getenv("IDLE_TIMEOUT_SECONDS", 120))
