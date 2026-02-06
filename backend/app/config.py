import os
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = BASE_DIR / "data"
EOS_MODELS_DIR = BASE_DIR / "models"

TRAINING_DATA_PATH = DATA_DIR / "training-data.json"

# CORS origins - configurable via environment variable for production
_default_origins = "http://localhost:3000,http://127.0.0.1:3000"
CORS_ORIGINS = os.environ.get("CORS_ORIGINS", _default_origins).split(",")
