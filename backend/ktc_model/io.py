"""Save and load model bundles (models, clip bounds, calibrators, metrics)."""

import json
from pathlib import Path

import joblib


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

    return {
        "models": models,
        "clip_bounds": clip_bounds,
        "calibrators": calibrators,
        "metrics": metrics,
    }
