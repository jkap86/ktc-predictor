"""Save and load model bundles (models, clip bounds, calibrators, metrics)."""

import json
from pathlib import Path

import joblib
import numpy as np


class EnsembleModel:
    """Wrapper that averages predictions from multiple models.

    This class is defined here in io.py to ensure it's importable during
    model loading (joblib needs access to the class definition).
    """

    def __init__(self, models: list):
        self.models = models

    def predict(self, X):
        preds = np.array([m.predict(X) for m in self.models])
        return np.mean(preds, axis=0)

    def fit(self, X, y):
        """Fit all models in the ensemble."""
        for m in self.models:
            m.fit(X, y)
        return self


def save_bundle(bundle: dict, out_dir: str) -> None:
    """Save a training bundle to disk.

    Writes:
      - <out_dir>/<POS>.joblib for each model
      - <out_dir>/clip_bounds.json
      - <out_dir>/calibrators/<POS>.joblib for each calibrator
      - <out_dir>/metrics.json

    Parameters
    ----------
    bundle : dict
        Keys: "models", "clip_bounds", "calibrators", "metrics"
    out_dir : str
        Directory to write files into.
    """
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)

    # Save models
    for pos, model in bundle["models"].items():
        joblib.dump(model, out / f"{pos}.joblib")

    # Save clip bounds as JSON (tuples -> lists)
    bounds_serializable = {
        pos: list(bounds) for pos, bounds in bundle["clip_bounds"].items()
    }
    with open(out / "clip_bounds.json", "w") as f:
        json.dump(bounds_serializable, f, indent=2)

    # Save calibrators
    cal_dir = out / "calibrators"
    cal_dir.mkdir(exist_ok=True)
    for pos, cal in bundle["calibrators"].items():
        if cal is not None:
            joblib.dump(cal, cal_dir / f"{pos}.joblib")

    # Save metrics
    if "metrics" in bundle and bundle["metrics"]:
        with open(out / "metrics.json", "w") as f:
            json.dump(bundle["metrics"], f, indent=2)

    # Save sentinel imputation values
    if "sentinel_impute" in bundle and bundle["sentinel_impute"]:
        with open(out / "sentinel_impute.json", "w") as f:
            json.dump(bundle["sentinel_impute"], f, indent=2)

    # Save quantile models for uncertainty estimation
    if "quantile_models" in bundle and bundle["quantile_models"]:
        q_dir = out / "quantile_models"
        q_dir.mkdir(exist_ok=True)
        for pos, q_models in bundle["quantile_models"].items():
            for quantile, model in q_models.items():
                # Convert float quantile to string for filename (e.g., 0.2 -> "p20")
                q_name = f"p{int(quantile * 100)}"
                joblib.dump(model, q_dir / f"{pos}_{q_name}.joblib")


def load_bundle(model_dir: str) -> dict:
    """Load a saved model bundle from disk.

    Parameters
    ----------
    model_dir : str
        Directory containing saved model files.

    Returns
    -------
    dict
        Keys: "models", "clip_bounds", "calibrators", "metrics"
    """
    d = Path(model_dir)

    models = {}
    for pos_file in d.glob("*.joblib"):
        pos = pos_file.stem
        models[pos] = joblib.load(pos_file)

    clip_bounds = {}
    bounds_path = d / "clip_bounds.json"
    if bounds_path.exists():
        with open(bounds_path) as f:
            raw = json.load(f)
        clip_bounds = {pos: tuple(bounds) for pos, bounds in raw.items()}

    calibrators = {}
    cal_dir = d / "calibrators"
    if cal_dir.exists():
        for cal_file in cal_dir.glob("*.joblib"):
            pos = cal_file.stem
            calibrators[pos] = joblib.load(cal_file)

    metrics = {}
    metrics_path = d / "metrics.json"
    if metrics_path.exists():
        with open(metrics_path) as f:
            metrics = json.load(f)

    sentinel_impute = {}
    sentinel_path = d / "sentinel_impute.json"
    if sentinel_path.exists():
        with open(sentinel_path) as f:
            sentinel_impute = json.load(f)

    residual_bands = {}
    bands_path = d / "residual_bands.json"
    if bands_path.exists():
        with open(bands_path) as f:
            residual_bands = json.load(f)

    # Load quantile models for uncertainty estimation
    quantile_models = {}
    q_dir = d / "quantile_models"
    if q_dir.exists():
        for q_file in q_dir.glob("*.joblib"):
            # Parse filename: e.g., "QB_p20.joblib" -> pos="QB", quantile=0.2
            parts = q_file.stem.split("_")
            if len(parts) >= 2:
                pos = parts[0]
                q_str = parts[1]  # e.g., "p20"
                if q_str.startswith("p"):
                    quantile = int(q_str[1:]) / 100  # "p20" -> 0.2
                    if pos not in quantile_models:
                        quantile_models[pos] = {}
                    quantile_models[pos][quantile] = joblib.load(q_file)

    return {
        "models": models,
        "clip_bounds": clip_bounds,
        "calibrators": calibrators,
        "quantile_models": quantile_models,
        "metrics": metrics,
        "sentinel_impute": sentinel_impute,
        "residual_bands": residual_bands,
    }
