import os
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = BASE_DIR / "data"
MODELS_DIR = BASE_DIR / "trained_models"

TRAINING_DATA_PATH = DATA_DIR / "training-data.json"
MODEL_PATH = MODELS_DIR / "ktc_model.joblib"
WEEKLY_MODEL_PATH = MODELS_DIR / "weekly_ktc_model.joblib"
ENSEMBLE_MODEL_PATH = MODELS_DIR / "ktc_ensemble.joblib"
XGBOOST_MODEL_PATH = MODELS_DIR / "ktc_xgboost.joblib"
XGB_CALIBRATED_BREAKOUT_MODEL_PATH = MODELS_DIR / "ktc_xgb_calibrated_breakout.joblib"

# CORS origins - configurable via environment variable for production
_default_origins = "http://localhost:3000,http://127.0.0.1:3000"
CORS_ORIGINS = os.environ.get("CORS_ORIGINS", _default_origins).split(",")
