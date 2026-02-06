"""CLI training script for per-position KTC prediction models.

Usage:
    cd backend
    python -m ktc_model.train --zip data/training-data.zip --out models
"""

import argparse
import sys

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.isotonic import IsotonicRegression
from sklearn.metrics import mean_absolute_error, r2_score
from sklearn.model_selection import GroupShuffleSplit

from .data import build_weekly_snapshot_df
from .io import save_bundle

POSITIONS = ["QB", "RB", "WR", "TE"]
FEATURES = [
    "games_played_so_far",
    "ppg_so_far",
    "start_ktc",
    "age",
    "weeks_missed_so_far",
    "draft_pick",
    "years_remaining",
]
MIN_SAMPLES = 100

NAN_CHECK_COLS = ["age", "draft_pick", "years_remaining"]


def print_diagnostics(df: pd.DataFrame, test_rows: list[dict]) -> None:
    """Print post-training diagnostics: NaN counts, quintile bias, per-position summary."""
    if not test_rows:
        print("No test rows collected — skipping diagnostics.\n")
        return

    # ── Table A: NaN counts per position ──────────────────────────────
    print("=" * 70)
    print("POST-TRAINING DIAGNOSTICS")
    print("=" * 70)
    print()
    print("Table A — NaN counts per feature (training data)")
    print("-" * 70)
    header = f"{'Position':<10}"
    for col in NAN_CHECK_COLS:
        header += f"  {col:>18}"
    header += f"  {'total_rows':>12}"
    print(header)

    for pos in POSITIONS:
        pos_df = df[df["position"] == pos]
        n = len(pos_df)
        row = f"{pos:<10}"
        for col in NAN_CHECK_COLS:
            nan_count = int(pos_df[col].isna().sum())
            pct = 100.0 * nan_count / n if n > 0 else 0
            row += f"  {nan_count:>7} ({pct:4.1f}%)"
        row += f"  {n:>12}"
        print(row)
    print()

    # ── Build test DataFrame ──────────────────────────────────────────
    test_df = pd.DataFrame(test_rows)
    test_df["signed_error"] = test_df["predicted_end_ktc"] - test_df["actual_end_ktc"]
    test_df["abs_error"] = test_df["signed_error"].abs()

    # ── Table B: Bias by start_ktc quintile ───────────────────────────
    print("Table B — Bias by start_ktc quintile")
    print("-" * 70)

    # Overall quintiles
    test_df["quintile"] = pd.qcut(test_df["start_ktc"], 5, labels=False, duplicates="drop") + 1
    quintile_ranges = test_df.groupby("quintile")["start_ktc"].agg(["min", "max"])

    print(f"{'Quintile':<10} {'Count':>6}  {'Bias':>8}  {'MAE':>8}  {'KTC Range':>18}")
    for q in sorted(test_df["quintile"].unique()):
        q_df = test_df[test_df["quintile"] == q]
        lo = quintile_ranges.loc[q, "min"]
        hi = quintile_ranges.loc[q, "max"]
        print(
            f"  Q{q:<7} {len(q_df):>6}  {q_df['signed_error'].mean():>+8.1f}  "
            f"{q_df['abs_error'].mean():>8.1f}  {lo:>7.0f} - {hi:>6.0f}"
        )
    print()

    # Per-position quintile breakdown
    for pos in POSITIONS:
        pos_test = test_df[test_df["position"] == pos]
        if pos_test.empty:
            continue
        pos_test = pos_test.copy()
        pos_test["pos_quintile"] = pd.qcut(
            pos_test["start_ktc"], 5, labels=False, duplicates="drop"
        ) + 1
        pos_ranges = pos_test.groupby("pos_quintile")["start_ktc"].agg(["min", "max"])
        print(f"  {pos}:")
        for q in sorted(pos_test["pos_quintile"].unique()):
            q_df = pos_test[pos_test["pos_quintile"] == q]
            lo = pos_ranges.loc[q, "min"]
            hi = pos_ranges.loc[q, "max"]
            print(
                f"    Q{q:<5} {len(q_df):>6}  {q_df['signed_error'].mean():>+8.1f}  "
                f"{q_df['abs_error'].mean():>8.1f}  {lo:>7.0f} - {hi:>6.0f}"
            )
        print()

    # ── Table C: Per-position summary ─────────────────────────────────
    print("Table C — Per-position test-set summary")
    print("-" * 70)
    print(f"{'Position':<10} {'Count':>6}  {'MAE':>8}  {'Bias':>8}")
    for pos in POSITIONS:
        pos_test = test_df[test_df["position"] == pos]
        if pos_test.empty:
            continue
        print(
            f"{pos:<10} {len(pos_test):>6}  {pos_test['abs_error'].mean():>8.1f}  "
            f"{pos_test['signed_error'].mean():>+8.1f}"
        )
    # Overall
    print(
        f"{'ALL':<10} {len(test_df):>6}  {test_df['abs_error'].mean():>8.1f}  "
        f"{test_df['signed_error'].mean():>+8.1f}"
    )
    print("=" * 70)
    print()


def _try_xgb(seed: int):
    """Attempt to create an XGBRegressor with monotonic constraints."""
    try:
        from xgboost import XGBRegressor

        return XGBRegressor(
            n_estimators=200,
            max_depth=5,
            learning_rate=0.1,
            monotone_constraints="(0,1,0,0,0,0,0)",
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
        monotonic_cst=[0, 1, 0, 0, 0, 0, 0],
        random_state=seed,
    )


def _monotonic_smoke_test(model, position: str) -> bool:
    """Verify log_ratio predictions are non-decreasing as PPG increases (gp=8, start_ktc=5000)."""
    ppg_values = [5, 10, 15, 20]
    X_test = np.array([[8, ppg, 5000, 25, 2, np.nan, 3] for ppg in ppg_values])
    preds = model.predict(X_test)

    is_monotonic = all(preds[i] <= preds[i + 1] for i in range(len(preds) - 1))
    status = "PASS" if is_monotonic else "FAIL"
    print(f"  Monotonic test ({position}): {status}  preds={[round(p, 4) for p in preds]}")
    return is_monotonic


def train_all(
    zip_path: str,
    json_name: str = "training-data.json",
    out_dir: str = "models",
    test_size: float = 0.2,
    seed: int = 42,
    no_calibration: bool = False,
    prefer_xgb: bool = False,
    export_csv: str | None = None,
) -> dict:
    """Train per-position models and return bundle."""
    print(f"Loading data from {zip_path}...")
    df = build_weekly_snapshot_df(zip_path, json_name)
    print(f"  Total rows: {len(df)}")
    print(f"  Positions: {df.groupby('position').size().to_dict()}")
    print()

    bundle = {"models": {}, "clip_bounds": {}, "calibrators": {}, "metrics": {}}
    all_test_rows: list[dict] = []

    for pos in POSITIONS:
        pos_df = df[df["position"] == pos].copy()
        n = len(pos_df)

        if n < MIN_SAMPLES:
            print(f"[{pos}] Skipping: only {n} samples (need >= {MIN_SAMPLES})")
            print()
            continue

        print(f"[{pos}] Training on {n} samples...")

        X = pos_df[FEATURES].values
        y = pos_df["log_ratio"].values
        y_end_ktc = pos_df["end_ktc"].values
        start_ktc = pos_df["start_ktc"].values
        groups = pos_df["player_id"].values

        # Group-based train/test split (no player leakage)
        gss = GroupShuffleSplit(n_splits=1, test_size=test_size, random_state=seed)
        train_idx, test_idx = next(gss.split(X, y, groups))

        X_train, X_test = X[train_idx], X[test_idx]
        y_train, y_test = y[train_idx], y[test_idx]
        start_ktc_test = start_ktc[test_idx]
        y_end_ktc_test = y_end_ktc[test_idx]

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

        # Clip test predictions (on log-ratio scale)
        test_preds_clipped = np.clip(test_preds, low, high)

        # Convert to end_ktc for reporting: predicted_end = start_ktc * exp(log_ratio)
        test_end_ktc_preds = start_ktc_test * np.exp(test_preds_clipped)

        # Collect test-set rows (always — used by diagnostics and optional CSV export)
        test_meta = pos_df.iloc[test_idx]
        for i in range(len(test_idx)):
            actual_delta = float(y_end_ktc_test[i] - start_ktc_test[i])
            predicted_delta = float(test_end_ktc_preds[i] - start_ktc_test[i])
            all_test_rows.append({
                "player_id": test_meta.iloc[i]["player_id"],
                "position": pos,
                "year": test_meta.iloc[i]["year"],
                "week": test_meta.iloc[i]["week"],
                "games_played": test_meta.iloc[i]["games_played_so_far"],
                "ppg": round(test_meta.iloc[i]["ppg_so_far"], 2),
                "start_ktc": start_ktc_test[i],
                "actual_end_ktc": y_end_ktc_test[i],
                "predicted_end_ktc": round(test_end_ktc_preds[i], 1),
                "actual_delta": round(actual_delta, 1),
                "predicted_delta": round(predicted_delta, 1),
                "actual_log_ratio": round(float(y_test[i]), 4),
                "predicted_log_ratio": round(float(test_preds_clipped[i]), 4),
                "error": round(test_end_ktc_preds[i] - y_end_ktc_test[i], 1),
            })

        mae = mean_absolute_error(y_end_ktc_test, test_end_ktc_preds)
        r2 = r2_score(y_end_ktc_test, test_end_ktc_preds)

        print(f"  Backend: {backend_name}")
        print(f"  n_train={len(X_train)}, n_test={len(X_test)}")
        print(f"  Clip bounds: [{low:.4f}, {high:.4f}]")
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

    if export_csv and all_test_rows:
        csv_df = pd.DataFrame(all_test_rows)
        csv_df = csv_df.sort_values("error", key=abs, ascending=False)
        csv_df.to_csv(export_csv, index=False)
        print(f"Exported {len(csv_df)} test-set predictions to {export_csv}")
        print()

    print_diagnostics(df, all_test_rows)

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
    parser.add_argument(
        "--export-csv",
        default=None,
        help="Export test-set predictions vs actuals to CSV",
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
        export_csv=args.export_csv,
    )


if __name__ == "__main__":
    main()
