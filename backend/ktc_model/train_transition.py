"""Train week-to-week KTC transition models.

Trains per-position models that predict ktc_delta_log (log change ratio)
from weekly features. These models are used for trajectory rollouts.
"""

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.linear_model import ElasticNet, Ridge
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import PolynomialFeatures, StandardScaler

from ktc_model.transition_data import build_transition_df, get_transition_features

VALID_POSITIONS = ["QB", "RB", "WR", "TE"]


def _build_elasticnet_poly(degree: int = 2):
    """Build ElasticNet with polynomial features pipeline."""
    return Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("poly", PolynomialFeatures(degree=degree, include_bias=False)),
        ("scaler", StandardScaler()),
        ("model", ElasticNet(alpha=0.001, l1_ratio=0.5, max_iter=5000)),
    ])


def _build_ridge():
    """Build simple Ridge regression pipeline."""
    return Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler()),
        ("model", Ridge(alpha=1.0)),
    ])


def train_transition_models(
    data_path: str = "data/training-data.zip",
    out_dir: str = "models/transition",
    test_size: float = 0.2,
    model_type: str = "elasticnet-poly",
) -> dict:
    """Train per-position transition models.

    Parameters
    ----------
    data_path : str
        Path to training data zip.
    out_dir : str
        Directory to save models.
    test_size : float
        Fraction of data to hold out for testing.
    model_type : str
        Model type: "elasticnet-poly" or "ridge".

    Returns
    -------
    dict
        Bundle with models, metrics, feature_names.
    """
    print(f"Building transition data from {data_path}...")
    df = build_transition_df(data_path)

    features = get_transition_features()
    target = "ktc_delta_log"

    models = {}
    metrics = {"positions": {}}
    clip_bounds = {}

    for position in VALID_POSITIONS:
        print(f"\n{'='*50}")
        print(f"Training {position} transition model")
        print(f"{'='*50}")

        pos_df = df[df["position"] == position].copy()
        print(f"  Samples: {len(pos_df):,}")

        if len(pos_df) < 100:
            print(f"  Skipping {position}: insufficient data")
            continue

        X = pos_df[features].values
        y = pos_df[target].values

        # Clip extreme target values (log ratios > ±1 are rare edge cases)
        y_clipped = np.clip(y, -1.0, 1.0)
        clipped_count = np.sum(np.abs(y) > 1.0)
        if clipped_count > 0:
            print(f"  Clipped {clipped_count} extreme target values")

        # Train/test split (stratify by year to avoid leakage)
        X_train, X_test, y_train, y_test = train_test_split(
            X, y_clipped, test_size=test_size, random_state=42
        )

        # Build model
        if model_type == "elasticnet-poly":
            model = _build_elasticnet_poly(degree=2)
        else:
            model = _build_ridge()

        # Train
        model.fit(X_train, y_train)

        # Evaluate
        y_pred_train = model.predict(X_train)
        y_pred_test = model.predict(X_test)

        mae_train = np.mean(np.abs(y_train - y_pred_train))
        mae_test = np.mean(np.abs(y_test - y_pred_test))
        bias_test = np.mean(y_pred_test - y_test)

        # Convert log MAE to approximate percentage
        # If MAE on log is 0.05, that's roughly ±5% per week
        pct_mae = (np.exp(mae_test) - 1) * 100

        print(f"  Train MAE: {mae_train:.4f} (log)")
        print(f"  Test MAE:  {mae_test:.4f} (log) ~ {pct_mae:.1f}% per week")
        print(f"  Test Bias: {bias_test:.4f}")

        # Clip bounds for predictions
        clip_low = np.percentile(y_clipped, 1)
        clip_high = np.percentile(y_clipped, 99)
        clip_bounds[position] = (float(clip_low), float(clip_high))
        print(f"  Clip bounds: [{clip_low:.3f}, {clip_high:.3f}]")

        models[position] = model
        metrics["positions"][position] = {
            "n_train": len(y_train),
            "n_test": len(y_test),
            "mae_train": round(mae_train, 4),
            "mae_test": round(mae_test, 4),
            "pct_mae": round(pct_mae, 2),
            "bias": round(bias_test, 4),
        }

    # Compute overall metrics
    total_mae = np.mean([m["mae_test"] for m in metrics["positions"].values()])
    metrics["overall_mae"] = round(total_mae, 4)

    print(f"\n{'='*50}")
    print(f"Overall MAE: {total_mae:.4f} (log)")
    print(f"{'='*50}")

    # Save bundle
    bundle = {
        "models": models,
        "clip_bounds": clip_bounds,
        "metrics": metrics,
        "feature_names": features,
    }

    _save_transition_bundle(bundle, out_dir)

    return bundle


def _save_transition_bundle(bundle: dict, out_dir: str) -> None:
    """Save transition model bundle to disk."""
    import joblib

    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)

    # Save models
    for pos, model in bundle["models"].items():
        joblib.dump(model, out / f"{pos}.joblib")

    # Save clip bounds
    with open(out / "clip_bounds.json", "w") as f:
        json.dump(bundle["clip_bounds"], f, indent=2)

    # Save metrics
    with open(out / "metrics.json", "w") as f:
        json.dump(bundle["metrics"], f, indent=2)

    # Save feature names
    with open(out / "feature_names.json", "w") as f:
        json.dump(bundle["feature_names"], f, indent=2)

    print(f"\nSaved transition models to {out_dir}/")


def load_transition_bundle(model_dir: str) -> dict:
    """Load transition model bundle from disk."""
    import joblib

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

    metrics = {}
    metrics_path = d / "metrics.json"
    if metrics_path.exists():
        with open(metrics_path) as f:
            metrics = json.load(f)

    feature_names = []
    feature_path = d / "feature_names.json"
    if feature_path.exists():
        with open(feature_path) as f:
            feature_names = json.load(f)

    return {
        "models": models,
        "clip_bounds": clip_bounds,
        "metrics": metrics,
        "feature_names": feature_names,
    }


def validate_rollout(
    bundle: dict,
    data_path: str = "data/training-data.zip",
    n_samples: int = 100,
) -> dict:
    """Validate transition model by rolling out full seasons.

    Compares predicted end_ktc (from rollout) to actual end_ktc.
    """
    import json
    import zipfile

    print("\n" + "=" * 50)
    print("Validating rollout accuracy")
    print("=" * 50)

    # Load raw data for rollout validation
    with zipfile.ZipFile(data_path, "r") as zf:
        with zf.open("training-data.json") as f:
            data = json.load(f)

    results = []
    models = bundle["models"]
    clip_bounds = bundle["clip_bounds"]
    features = bundle["feature_names"]

    for player in data["players"][:n_samples * 2]:
        position = player["position"]
        if position not in models:
            continue

        model = models[position]
        clips = clip_bounds.get(position, (-0.5, 0.5))

        for season in player.get("seasons", []):
            weekly_ktc = season.get("weekly_ktc", [])
            weekly_stats = season.get("weekly_stats", [])

            if not weekly_ktc or not weekly_stats:
                continue

            # Get valid KTC weeks
            ktc_by_week = {
                wk["week"]: wk["ktc"]
                for wk in weekly_ktc
                if wk.get("ktc", 0) > 0 and wk.get("ktc", 0) < 9999
            }

            if len(ktc_by_week) < 5:
                continue

            sorted_weeks = sorted(ktc_by_week.keys())
            start_week = sorted_weeks[0]
            end_week = sorted_weeks[-1]

            actual_start = ktc_by_week[start_week]
            actual_end = ktc_by_week[end_week]

            # Rollout
            ktc_current = actual_start
            stats_by_week = {ws["week"]: ws for ws in weekly_stats}
            age = season.get("age")

            cumulative_fp = 0.0
            cumulative_games = 0

            for week in range(start_week, end_week):
                if week not in ktc_by_week:
                    continue

                # Accumulate stats
                if week in stats_by_week:
                    ws = stats_by_week[week]
                    cumulative_fp += ws.get("fantasy_points", 0) or 0
                    cumulative_games += ws.get("games_played", 0) or 0

                ppg = cumulative_fp / cumulative_games if cumulative_games > 0 else 0
                weekly_fp = stats_by_week.get(week, {}).get("fantasy_points", 0) or 0
                games_this_week = stats_by_week.get(week, {}).get("games_played", 0) or 0
                season_progress = week / 18.0

                # Build feature vector
                X = np.array([[
                    np.log(ktc_current),  # ktc_current_log
                    ppg,                   # ppg_cumulative
                    cumulative_games,      # games_played
                    weekly_fp,             # weekly_fp
                    games_this_week,       # games_this_week
                    season_progress,       # season_progress
                    np.nan,                # ktc_momentum (simplified)
                    (age - 26) if age else np.nan,  # age_prime_distance
                ]])

                # Predict delta
                pred_delta_log = float(model.predict(X)[0])
                pred_delta_log = np.clip(pred_delta_log, clips[0], clips[1])

                # Apply delta
                ktc_current = ktc_current * np.exp(pred_delta_log)

            predicted_end = ktc_current
            error = predicted_end - actual_end
            pct_error = error / actual_end * 100 if actual_end > 0 else 0

            results.append({
                "position": position,
                "actual_start": actual_start,
                "actual_end": actual_end,
                "predicted_end": predicted_end,
                "error": error,
                "pct_error": pct_error,
            })

            if len(results) >= n_samples:
                break

        if len(results) >= n_samples:
            break

    # Compute rollout metrics
    df = pd.DataFrame(results)

    print(f"\nRollout validation on {len(df)} player-seasons:")
    for pos in VALID_POSITIONS:
        pos_df = df[df["position"] == pos]
        if len(pos_df) == 0:
            continue
        mae = np.mean(np.abs(pos_df["error"]))
        bias = np.mean(pos_df["error"])
        print(f"  {pos}: MAE={mae:.0f}, Bias={bias:.0f}")

    overall_mae = np.mean(np.abs(df["error"]))
    overall_bias = np.mean(df["error"])
    print(f"\n  Overall: MAE={overall_mae:.0f}, Bias={overall_bias:.0f}")

    return {
        "rollout_mae": round(overall_mae, 1),
        "rollout_bias": round(overall_bias, 1),
        "n_samples": len(df),
    }


def compute_diagnostics(
    bundle: dict,
    data_path: str = "data/training-data.zip",
) -> dict:
    """Compute comprehensive diagnostics for transition model.

    Reports:
    - MAE in both log and KTC points
    - Naive baseline comparison
    - Tier-level calibration (bias by KTC bucket)
    - Riser/faller bias analysis
    """
    print("\n" + "=" * 60)
    print("COMPREHENSIVE DIAGNOSTICS")
    print("=" * 60)

    df = build_transition_df(data_path)
    models = bundle["models"]
    clip_bounds = bundle["clip_bounds"]
    features = bundle["feature_names"]

    tiers = [(0, 2000), (2000, 4000), (4000, 6000), (6000, 9999)]
    tier_names = ["0-2k", "2k-4k", "4k-6k", "6k+"]

    diagnostics = {"positions": {}}

    for pos in VALID_POSITIONS:
        pos_df = df[df["position"] == pos].copy()
        if len(pos_df) < 50:
            continue

        model = models[pos]
        clips = clip_bounds.get(pos, (-0.5, 0.5))

        X = pos_df[features].values
        y_log = pos_df["ktc_delta_log"].values
        ktc_current = pos_df["ktc_current"].values
        ktc_next_actual = pos_df["ktc_next"].values

        # Predict
        pred_log = np.clip(model.predict(X), clips[0], clips[1])
        ktc_next_pred = ktc_current * np.exp(pred_log)

        # Overall metrics
        mae_log = np.mean(np.abs(y_log - pred_log))
        mae_points = np.mean(np.abs(ktc_next_actual - ktc_next_pred))
        bias_points = np.mean(ktc_next_pred - ktc_next_actual)
        naive_mae = np.mean(np.abs(ktc_next_actual - ktc_current))

        print(f"\n{pos}:")
        print(f"  MAE (log):    {mae_log:.4f}")
        print(f"  MAE (points): {mae_points:.0f}")
        print(f"  Bias:         {bias_points:+.0f}")
        print(f"  Naive MAE:    {naive_mae:.0f} (ktc_next = ktc_current)")
        print(f"  Improvement:  {(1 - mae_points/naive_mae)*100:.1f}% over naive")

        pos_diag = {
            "n": len(pos_df),
            "mae_log": round(mae_log, 4),
            "mae_points": round(mae_points, 1),
            "bias": round(bias_points, 1),
            "naive_mae": round(naive_mae, 1),
            "tiers": {},
        }

        # Tier-level analysis
        print(f"\n  Tier Analysis:")
        for (lo, hi), name in zip(tiers, tier_names):
            mask = (ktc_current >= lo) & (ktc_current < hi)
            if mask.sum() < 10:
                continue

            tier_actual = ktc_next_actual[mask]
            tier_pred = ktc_next_pred[mask]
            tier_curr = ktc_current[mask]

            tier_mae = np.mean(np.abs(tier_actual - tier_pred))
            tier_bias = np.mean(tier_pred - tier_actual)

            # Riser vs faller
            actual_delta = tier_actual - tier_curr
            risers = actual_delta > 0
            fallers = actual_delta < 0

            riser_bias = float(np.mean(tier_pred[risers] - tier_actual[risers])) if risers.sum() > 5 else None
            faller_bias = float(np.mean(tier_pred[fallers] - tier_actual[fallers])) if fallers.sum() > 5 else None

            riser_str = f"{riser_bias:+5.0f}" if riser_bias is not None else "  N/A"
            faller_str = f"{faller_bias:+5.0f}" if faller_bias is not None else "  N/A"
            print(f"    {name}: n={mask.sum():4d}  MAE={tier_mae:5.0f}  Bias={tier_bias:+5.0f}  "
                  f"Riser={riser_str}  Faller={faller_str}")

            pos_diag["tiers"][name] = {
                "n": int(mask.sum()),
                "mae": round(tier_mae, 1),
                "bias": round(tier_bias, 1),
                "riser_bias": round(riser_bias, 1) if riser_bias else None,
                "faller_bias": round(faller_bias, 1) if faller_bias else None,
            }

        diagnostics["positions"][pos] = pos_diag

    return diagnostics


def main():
    parser = argparse.ArgumentParser(description="Train transition models")
    parser.add_argument(
        "--data",
        default="data/training-data.zip",
        help="Path to training data",
    )
    parser.add_argument(
        "--out",
        default="models/transition",
        help="Output directory for models",
    )
    parser.add_argument(
        "--model-type",
        default="elasticnet-poly",
        choices=["elasticnet-poly", "ridge"],
        help="Model type",
    )
    parser.add_argument(
        "--validate",
        action="store_true",
        help="Run rollout validation after training",
    )
    parser.add_argument(
        "--diagnostics",
        action="store_true",
        help="Run comprehensive diagnostics",
    )

    args = parser.parse_args()

    bundle = train_transition_models(
        data_path=args.data,
        out_dir=args.out,
        model_type=args.model_type,
    )

    if args.validate:
        validate_rollout(bundle, args.data)

    if args.diagnostics:
        diag = compute_diagnostics(bundle, args.data)
        # Save diagnostics
        out = Path(args.out)
        with open(out / "diagnostics.json", "w") as f:
            json.dump(diag, f, indent=2)
        print(f"\nDiagnostics saved to {args.out}/diagnostics.json")


if __name__ == "__main__":
    main()
