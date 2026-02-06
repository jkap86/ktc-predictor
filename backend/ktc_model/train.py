"""CLI training script for per-position KTC prediction models.

Usage:
    cd backend
    python -m ktc_model.train --zip data/training-data.zip --out models
"""

import argparse
import sys

import numpy as np
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.isotonic import IsotonicRegression
from sklearn.metrics import mean_absolute_error, r2_score
from sklearn.model_selection import GroupShuffleSplit

from .data import build_weekly_snapshot_df
from .io import save_bundle

POSITIONS = ["QB", "RB", "WR", "TE"]
FEATURES = ["games_played_so_far", "ppg_so_far"]
MIN_SAMPLES = 100


def _try_xgb(seed: int):
    """Attempt to create an XGBRegressor with monotonic constraints."""
    try:
        from xgboost import XGBRegressor

        return XGBRegressor(
            n_estimators=200,
            max_depth=5,
            learning_rate=0.1,
            monotone_constraints="(0,1)",
            random_state=seed,
        )
    except ImportError:
        return None


def _build_hgb(seed: int):
    """Create a HistGradientBoostingRegressor with monotonic constraints."""
    return HistGradientBoostingRegressor(
        max_iter=200,
        max_depth=5,
        learning_rate=0.1,
        monotonic_cst=[0, 1],
        random_state=seed,
    )


def _monotonic_smoke_test(model, position: str) -> bool:
    """Verify predictions are non-decreasing as PPG increases (gp=8)."""
    ppg_values = [5, 10, 15, 20]
    X_test = np.array([[8, ppg] for ppg in ppg_values])
    preds = model.predict(X_test)

    is_monotonic = all(preds[i] <= preds[i + 1] for i in range(len(preds) - 1))
    status = "PASS" if is_monotonic else "FAIL"
    print(f"  Monotonic test ({position}): {status}  preds={[round(p, 1) for p in preds]}")
    return is_monotonic


def train_all(
    zip_path: str,
    json_name: str = "training-data.json",
    out_dir: str = "models",
    test_size: float = 0.2,
    seed: int = 42,
    no_calibration: bool = False,
    prefer_xgb: bool = False,
) -> dict:
    """Train per-position models and return bundle."""
    print(f"Loading data from {zip_path}...")
    df = build_weekly_snapshot_df(zip_path, json_name)
    print(f"  Total rows: {len(df)}")
    print(f"  Positions: {df.groupby('position').size().to_dict()}")
    print()

    bundle = {"models": {}, "clip_bounds": {}, "calibrators": {}, "metrics": {}}

    for pos in POSITIONS:
        pos_df = df[df["position"] == pos].copy()
        n = len(pos_df)

        if n < MIN_SAMPLES:
            print(f"[{pos}] Skipping: only {n} samples (need >= {MIN_SAMPLES})")
            print()
            continue

        print(f"[{pos}] Training on {n} samples...")

        X = pos_df[FEATURES].values
        y = pos_df["end_ktc"].values
        groups = pos_df["player_id"].values

        # Group-based train/test split (no player leakage)
        gss = GroupShuffleSplit(n_splits=1, test_size=test_size, random_state=seed)
        train_idx, test_idx = next(gss.split(X, y, groups))

        X_train, X_test = X[train_idx], X[test_idx]
        y_train, y_test = y[train_idx], y[test_idx]

        # Select backend
        backend_name = "HGB"
        if prefer_xgb:
            model = _try_xgb(seed)
            if model is not None:
                backend_name = "XGB"
            else:
                print("  XGBoost not available, falling back to HGB")
                model = _build_hgb(seed)
        else:
            model = _build_hgb(seed)

        model.fit(X_train, y_train)

        # Monotonic smoke test
        _monotonic_smoke_test(model, pos)

        # Raw predictions
        train_preds = model.predict(X_train)
        test_preds = model.predict(X_test)

        # Isotonic calibration
        calibrator = None
        if not no_calibration:
            calibrator = IsotonicRegression(out_of_bounds="clip")
            calibrator.fit(train_preds, y_train)

            # Calibrate test predictions
            test_preds_cal = calibrator.predict(test_preds)
            nan_mask = ~np.isnan(test_preds_cal)
            if nan_mask.any():
                test_preds = np.where(nan_mask, test_preds_cal, test_preds)

        # Clip bounds: 2nd/98th percentile of y_train
        low = float(np.percentile(y_train, 2))
        high = float(np.percentile(y_train, 98))

        # Clip test predictions for metrics
        test_preds_clipped = np.clip(test_preds, low, high)

        mae = mean_absolute_error(y_test, test_preds_clipped)
        r2 = r2_score(y_test, test_preds_clipped)

        print(f"  Backend: {backend_name}")
        print(f"  n_train={len(X_train)}, n_test={len(X_test)}")
        print(f"  Clip bounds: [{low:.0f}, {high:.0f}]")
        print(f"  MAE: {mae:.1f}")
        print(f"  R²:  {r2:.4f}")
        print()

        bundle["models"][pos] = model
        bundle["clip_bounds"][pos] = (low, high)
        bundle["calibrators"][pos] = calibrator
        bundle["metrics"][pos] = {
            "mae": round(mae, 1),
            "r2": round(r2, 4),
            "n_train": len(X_train),
            "n_test": len(X_test),
            "backend": backend_name,
            "clip_low": round(low, 1),
            "clip_high": round(high, 1),
        }

    if bundle["models"]:
        save_bundle(bundle, out_dir)
        print(f"Saved {len(bundle['models'])} models to {out_dir}/")
    else:
        print("No models trained!")
        sys.exit(1)

    return bundle


def main():
    parser = argparse.ArgumentParser(
        description="Train per-position KTC prediction models"
    )
    parser.add_argument(
        "--zip",
        default="data/training-data.zip",
        help="Path to training data zip file",
    )
    parser.add_argument(
        "--json",
        default="training-data.json",
        help="Name of JSON file inside the zip",
    )
    parser.add_argument(
        "--out",
        default="models",
        help="Output directory for saved models",
    )
    parser.add_argument(
        "--test-size",
        type=float,
        default=0.2,
        help="Fraction of data for test set (default: 0.2)",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed (default: 42)",
    )
    parser.add_argument(
        "--no-calibration",
        action="store_true",
        help="Skip isotonic calibration",
    )
    parser.add_argument(
        "--prefer-xgb",
        action="store_true",
        help="Prefer XGBoost over HistGradientBoosting",
    )

    args = parser.parse_args()
    train_all(
        zip_path=args.zip,
        json_name=args.json,
        out_dir=args.out,
        test_size=args.test_size,
        seed=args.seed,
        no_calibration=args.no_calibration,
        prefer_xgb=args.prefer_xgb,
    )


if __name__ == "__main__":
    main()
