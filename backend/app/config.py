import os
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = BASE_DIR / "data"
EOS_MODELS_DIR = BASE_DIR / "models"
EOS_MODEL_VERSION = "eos_hgb_v1"

# Transition model directory
TRANSITION_MODELS_DIR = BASE_DIR / "models" / "transition"
TRANSITION_MODEL_VERSION = "transition_v1"

TRAINING_DATA_PATH = DATA_DIR / "training-data.json"

# CORS origins - configurable via environment variable for production
_default_origins = "http://localhost:3000,http://127.0.0.1:3000"
CORS_ORIGINS = os.environ.get("CORS_ORIGINS", _default_origins).split(",")
