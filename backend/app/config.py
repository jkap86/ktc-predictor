import os
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = BASE_DIR / "data"
ITERATIONS_DIR = BASE_DIR / "iterations"

TRAINING_DATA_PATH = DATA_DIR / "training-data.json"

# Default model iteration (can be overridden via env var)
DEFAULT_MODEL_ID = os.environ.get("KTC_DEFAULT_MODEL", "v4_hgb_momentum")

# Sleeper projection season (bump annually)
PROJECTION_SEASON = int(os.environ.get("KTC_PROJECTION_SEASON", "2026"))

# CORS origins - configurable via environment variable for production
_default_origins = "http://localhost:3002,http://127.0.0.1:3002"
CORS_ORIGINS = [o.strip() for o in os.environ.get("CORS_ORIGINS", _default_origins).split(",")]
