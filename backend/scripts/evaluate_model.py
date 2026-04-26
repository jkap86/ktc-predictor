"""Model evaluation script: baselines, per-bucket MAE/bias, worst misses, alternative backends.

Usage:
    cd backend
    python -m scripts.evaluate_model
    python -m scripts.evaluate_model --compare-backends   # also test XGBoost/LightGBM/CatBoost
"""

import argparse
import json
import sys
import zipfile
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, r2_score
from sklearn.model_selection import GroupShuffleSplit, GroupKFold
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.impute import SimpleImputer

POSITIONS = ["QB", "RB", "WR", "TE"]
KTC_BUCKETS = [(0, 1000), (1000, 3000), (3000, 6000), (6000, 9000), (9000, 10000)]
AGE_BUCKETS = {"QB": [(0, 26), (26, 30), (30, 34), (34, 99)],
               "RB": [(0, 23), (23, 26), (26, 28), (28, 99)],
               "WR": [(0, 24), (24, 27), (27, 30), (30, 99)],
               "TE": [(0, 25), (25, 28), (28, 31), (31, 99)]}


def _bucket_label(lo, hi):
    if hi >= 9999:
        return f"{lo}+"
    return f"{lo}-{hi}"


# ── Load test predictions from CSV ─────────────────────────────────────

def load_test_predictions(model_dir: str) -> pd.DataFrame:
    csv_path = Path(model_dir) / "test_predictions.csv"
    if not csv_path.exists():
        print(f"ERROR: {csv_path} not found. Re-train with --export-csv.")
        sys.exit(1)
    df = pd.read_csv(csv_path)
    df["signed_error"] = df["predicted_end_ktc"] - df["actual_end_ktc"]
    df["abs_error"] = df["signed_error"].abs()
    # Baseline: predicted = current KTC (no change)
    df["baseline_error"] = (df["start_ktc"] - df["actual_end_ktc"]).abs()
    df["baseline_signed"] = df["start_ktc"] - df["actual_end_ktc"]
    return df


# ── Baseline comparison ────────────────────────────────────────────────

def print_baseline_comparison(df: pd.DataFrame):
    print("=" * 80)
    print("BASELINE COMPARISON: Model vs 'No Change' (predicted = current KTC)")
    print("=" * 80)
    print()

    header = f"{'Position':<10} {'N':>6}  {'Model MAE':>10} {'Baseline MAE':>13} {'Delta':>8} {'Better?':>8}"
    print(header)
    print("-" * 70)

    for pos in POSITIONS:
        p = df[df["position"] == pos]
        if p.empty:
            continue
        m_mae = p["abs_error"].mean()
        b_mae = p["baseline_error"].mean()
        delta = m_mae - b_mae
        better = "YES" if delta < 0 else "NO"
        print(f"{pos:<10} {len(p):>6}  {m_mae:>10.1f} {b_mae:>13.1f} {delta:>+8.1f} {better:>8}")

    # Overall
    m_mae = df["abs_error"].mean()
    b_mae = df["baseline_error"].mean()
    delta = m_mae - b_mae
    better = "YES" if delta < 0 else "NO"
    print("-" * 70)
    print(f"{'ALL':<10} {len(df):>6}  {m_mae:>10.1f} {b_mae:>13.1f} {delta:>+8.1f} {better:>8}")
    print()

    # Per-position direction accuracy
    print("Direction accuracy (did model predict gain/loss correctly?):")
    print(f"{'Position':<10} {'Model':>8} {'Baseline':>10}")
    print("-" * 35)
    for pos in POSITIONS:
        p = df[df["position"] == pos]
        if p.empty:
            continue
        actual_dir = np.sign(p["actual_end_ktc"].values - p["start_ktc"].values)
        model_dir = np.sign(p["predicted_end_ktc"].values - p["start_ktc"].values)
        model_acc = (actual_dir == model_dir).mean()
        # Baseline always predicts 0 change, so direction is always 0
        # Count how often "no change" is correct (actual didn't change)
        baseline_acc = (actual_dir == 0).mean()
        print(f"{pos:<10} {model_acc:>7.1%} {baseline_acc:>9.1%}")
    print()


# ── MAE by KTC bucket ─────────────────────────────────────────────────

def print_mae_by_bucket(df: pd.DataFrame):
    print("=" * 80)
    print("MAE BY START KTC BUCKET (per position)")
    print("=" * 80)
    print()

    for pos in POSITIONS:
        p = df[df["position"] == pos]
        if p.empty:
            continue
        print(f"  {pos}:")
        print(f"    {'Bucket':<12} {'N':>5} {'Model MAE':>10} {'Baseline':>10} {'Bias':>8} {'Dir Acc':>8}")
        print(f"    {'-'*58}")
        for lo, hi in KTC_BUCKETS:
            bucket = p[(p["start_ktc"] >= lo) & (p["start_ktc"] < hi)]
            if len(bucket) < 5:
                continue
            m_mae = bucket["abs_error"].mean()
            b_mae = bucket["baseline_error"].mean()
            bias = bucket["signed_error"].mean()
            actual_dir = np.sign(bucket["actual_end_ktc"].values - bucket["start_ktc"].values)
            model_dir = np.sign(bucket["predicted_end_ktc"].values - bucket["start_ktc"].values)
            dir_acc = (actual_dir == model_dir).mean()
            label = _bucket_label(lo, hi)
            print(f"    {label:<12} {len(bucket):>5} {m_mae:>10.1f} {b_mae:>10.1f} {bias:>+8.1f} {dir_acc:>7.1%}")
        print()


# ── MAE by age bucket ──────────────────────────────────────────────────

def print_mae_by_age(df: pd.DataFrame):
    print("=" * 80)
    print("MAE BY AGE BUCKET (per position)")
    print("=" * 80)
    print()

    for pos in POSITIONS:
        p = df[df["position"] == pos].dropna(subset=["age"])
        if p.empty:
            continue
        buckets = AGE_BUCKETS[pos]
        print(f"  {pos}:")
        print(f"    {'Age':<12} {'N':>5} {'Model MAE':>10} {'Baseline':>10} {'Bias':>8}")
        print(f"    {'-'*50}")
        for lo, hi in buckets:
            bucket = p[(p["age"] >= lo) & (p["age"] < hi)]
            if len(bucket) < 5:
                continue
            m_mae = bucket["abs_error"].mean()
            b_mae = bucket["baseline_error"].mean()
            bias = bucket["signed_error"].mean()
            label = f"{lo}-{hi}" if hi < 99 else f"{lo}+"
            print(f"    {label:<12} {len(bucket):>5} {m_mae:>10.1f} {b_mae:>10.1f} {bias:>+8.1f}")
        print()


# ── Worst misses ───────────────────────────────────────────────────────

def print_worst_misses(df: pd.DataFrame, n: int = 25):
    print("=" * 80)
    print(f"TOP {n} WORST MISSES")
    print("=" * 80)
    print()

    # De-duplicate: for players with multiple weekly snapshots, keep the one
    # closest to the final week (most informative prediction window)
    deduped = df.sort_values("week", ascending=False).drop_duplicates(
        subset=["player_id", "position", "year"], keep="first"
    )
    worst = deduped.nlargest(n, "abs_error")

    print(f"{'PID':<8} {'Pos':<4} {'Year':>4} {'Age':>4} {'GP':>3} {'PPG':>6} "
          f"{'Start':>6} {'Actual':>7} {'Predicted':>9} {'Error':>7} {'BL Err':>7}")
    print("-" * 85)
    for _, row in worst.iterrows():
        age_str = f"{row['age']:.0f}" if pd.notna(row['age']) else "?"
        bl_err = abs(row["start_ktc"] - row["actual_end_ktc"])
        print(f"{row['player_id']:<8} {row['position']:<4} {row['year']:>4} {age_str:>4} "
              f"{row['games_played']:>3.0f} {row['ppg']:>6.1f} {row['start_ktc']:>6.0f} "
              f"{row['actual_end_ktc']:>7.0f} {row['predicted_end_ktc']:>9.0f} "
              f"{row['error']:>+7.0f} {bl_err:>7.0f}")
    print()

    # Categorize misses
    print("Miss patterns:")
    over = worst[worst["error"] > 0]
    under = worst[worst["error"] < 0]
    print(f"  Overpredictions: {len(over)}")
    print(f"  Underpredictions: {len(under)}")

    if len(over) > 0:
        print(f"  Avg overprediction start_ktc: {over['start_ktc'].mean():.0f}")
    if len(under) > 0:
        print(f"  Avg underprediction start_ktc: {under['start_ktc'].mean():.0f}")
    print()


# ── Best predictions ───────────────────────────────────────────────────

def print_best_predictions(df: pd.DataFrame, n: int = 15):
    print("=" * 80)
    print(f"TOP {n} BEST PREDICTIONS (where model beat baseline most)")
    print("=" * 80)
    print()

    df_copy = df.copy()
    df_copy["improvement"] = df_copy["baseline_error"] - df_copy["abs_error"]
    deduped = df_copy.sort_values("week", ascending=False).drop_duplicates(
        subset=["player_id", "position", "year"], keep="first"
    )
    best = deduped.nlargest(n, "improvement")

    print(f"{'PID':<8} {'Pos':<4} {'Year':>4} {'Start':>6} {'Actual':>7} {'Predicted':>9} "
          f"{'M Err':>6} {'BL Err':>7} {'Saved':>6}")
    print("-" * 75)
    for _, row in best.iterrows():
        print(f"{row['player_id']:<8} {row['position']:<4} {row['year']:>4} {row['start_ktc']:>6.0f} "
              f"{row['actual_end_ktc']:>7.0f} {row['predicted_end_ktc']:>9.0f} "
              f"{row['abs_error']:>6.0f} {row['baseline_error']:>7.0f} {row['improvement']:>+6.0f}")
    print()


# ── Alternative backend comparison ─────────────────────────────────────

def compare_backends(zip_path: str, json_name: str = "training-data.json"):
    """Train HGB, XGBoost, LightGBM, CatBoost on same splits and compare MAE."""
    print("=" * 80)
    print("BACKEND COMPARISON: HGB vs XGBoost vs LightGBM vs CatBoost")
    print("=" * 80)
    print()

    # Import v4 data builder (applies all patches)
    import iterations.v4_hgb_momentum.train  # noqa: F401 — patches v1
    import iterations.v1_hgb_baseline.train as v1

    print("Loading data...")
    df = v1.build_weekly_snapshot_df(zip_path, json_name)
    print(f"  {len(df)} rows loaded")
    print()

    backends = _get_available_backends()
    print(f"Available backends: {', '.join(backends.keys())}")
    print()

    results = []

    for pos in POSITIONS:
        pos_df = df[df["position"] == pos].copy()
        if len(pos_df) < 100:
            continue

        pos_features = v1.get_features_for_position(pos)
        X = pos_df[pos_features].values
        y = pos_df["log_ratio"].values
        start_ktc = pos_df["start_ktc"].values
        groups = pos_df["player_id"].values

        # Same split as training
        gss = GroupShuffleSplit(n_splits=1, test_size=0.2, random_state=42)
        train_idx, test_idx = next(gss.split(X, y, groups))
        X_train, X_test = X[train_idx], X[test_idx]
        y_train, y_test = y[train_idx], y[test_idx]
        start_ktc_test = start_ktc[test_idx]
        y_end_ktc_test = pos_df["end_ktc"].values[test_idx]

        # Impute for backends that don't handle NaN
        imputer = SimpleImputer(strategy="median")
        X_train_imp = imputer.fit_transform(X_train)
        X_test_imp = imputer.transform(X_test)

        # Sample weights (same as training)
        start_ktc_train = start_ktc[train_idx]
        sample_weights = np.ones(len(X_train))
        if pos in ("WR", "RB"):
            sample_weights[start_ktc_train >= 6000] = 1.5
            sample_weights[start_ktc_train >= 4000] = np.maximum(
                sample_weights[start_ktc_train >= 4000], 1.25)
        if pos == "RB":
            age_train = pos_df.iloc[train_idx]["age"].values
            sample_weights[(age_train <= 23) & (start_ktc_train >= 3000)] = np.maximum(
                sample_weights[(age_train <= 23) & (start_ktc_train >= 3000)], 2.0)
            sample_weights[(age_train <= 23) & (start_ktc_train >= 5000)] = np.maximum(
                sample_weights[(age_train <= 23) & (start_ktc_train >= 5000)], 2.5)

        # Baseline
        baseline_mae = mean_absolute_error(y_end_ktc_test, start_ktc_test)

        print(f"[{pos}] n_train={len(X_train)}, n_test={len(X_test)}, baseline_mae={baseline_mae:.1f}")

        for name, builder in backends.items():
            try:
                model = builder(pos)
                # HGB handles NaN natively; others need imputed data
                if name == "HGB":
                    model.fit(X_train, y_train, sample_weight=sample_weights)
                    preds = model.predict(X_test)
                elif name in ("XGBoost",):
                    model.fit(X_train_imp, y_train, sample_weight=sample_weights)
                    preds = model.predict(X_test_imp)
                else:
                    # LightGBM and CatBoost handle NaN
                    if name == "LightGBM":
                        model.fit(X_train, y_train, sample_weight=sample_weights)
                    elif name == "CatBoost":
                        model.fit(X_train, y_train, sample_weight=sample_weights, verbose=False)
                    preds = model.predict(X_test)

                pred_end_ktc = start_ktc_test * np.exp(np.clip(preds, -1.5, 2.0))
                mae = mean_absolute_error(y_end_ktc_test, pred_end_ktc)
                r2 = r2_score(y_end_ktc_test, pred_end_ktc)
                bias = float(np.mean(pred_end_ktc - y_end_ktc_test))

                results.append({
                    "position": pos, "backend": name,
                    "mae": mae, "r2": r2, "bias": bias,
                    "baseline_mae": baseline_mae,
                })
                print(f"  {name:<12} MAE={mae:>7.1f}  R²={r2:.4f}  bias={bias:>+7.1f}  vs_baseline={mae - baseline_mae:>+7.1f}")
            except Exception as e:
                print(f"  {name:<12} FAILED: {e}")
        print()

    # Summary table
    if results:
        print("=" * 80)
        print("SUMMARY: MAE by Position × Backend")
        print("=" * 80)
        rdf = pd.DataFrame(results)
        pivot = rdf.pivot_table(index="position", columns="backend", values="mae", aggfunc="first")
        # Add baseline column
        bl = rdf.drop_duplicates("position").set_index("position")["baseline_mae"]
        pivot.insert(0, "Baseline", bl)
        print(pivot.round(1).to_string())
        print()

        # Best per position
        print("Best backend per position:")
        for pos in POSITIONS:
            pos_results = [r for r in results if r["position"] == pos]
            if pos_results:
                best = min(pos_results, key=lambda r: r["mae"])
                print(f"  {pos}: {best['backend']} (MAE={best['mae']:.1f})")
        print()


def _get_available_backends() -> dict:
    """Return dict of available ML backends."""
    from iterations.v1_hgb_baseline.train import POSITION_HYPERPARAMS

    backends = {}

    # HGB (always available)
    def build_hgb(pos):
        params = POSITION_HYPERPARAMS.get(pos, POSITION_HYPERPARAMS["WR"])
        return HistGradientBoostingRegressor(
            max_iter=params["n_estimators"],
            max_depth=params["max_depth"],
            learning_rate=params["learning_rate"],
            loss=params.get("loss", "absolute_error"),
            min_samples_leaf=params.get("min_samples_leaf", 20),
            l2_regularization=params.get("l2_regularization", 0.0),
            max_leaf_nodes=params.get("max_leaf_nodes", 31),
            random_state=42,
        )
    backends["HGB"] = build_hgb

    # XGBoost
    try:
        from xgboost import XGBRegressor
        def build_xgb(pos):
            params = POSITION_HYPERPARAMS.get(pos, POSITION_HYPERPARAMS["WR"])
            return XGBRegressor(
                n_estimators=params["n_estimators"],
                max_depth=params["max_depth"],
                learning_rate=params["learning_rate"],
                random_state=42,
                n_jobs=-1,
            )
        backends["XGBoost"] = build_xgb
    except ImportError:
        pass

    # LightGBM
    try:
        import lightgbm as lgb
        def build_lgbm(pos):
            params = POSITION_HYPERPARAMS.get(pos, POSITION_HYPERPARAMS["WR"])
            return lgb.LGBMRegressor(
                n_estimators=params["n_estimators"],
                max_depth=params["max_depth"],
                learning_rate=params["learning_rate"],
                random_state=42,
                n_jobs=-1,
                verbose=-1,
            )
        backends["LightGBM"] = build_lgbm
    except ImportError:
        pass

    # CatBoost
    try:
        from catboost import CatBoostRegressor
        def build_catboost(pos):
            params = POSITION_HYPERPARAMS.get(pos, POSITION_HYPERPARAMS["WR"])
            return CatBoostRegressor(
                iterations=params["n_estimators"],
                depth=min(params["max_depth"], 10),
                learning_rate=params["learning_rate"],
                random_seed=42,
                verbose=0,
            )
        backends["CatBoost"] = build_catboost
    except ImportError:
        pass

    return backends


# ── Cross-validated backend comparison ─────────────────────────────────

def compare_backends_cv(zip_path: str, json_name: str = "training-data.json", n_folds: int = 3):
    """Cross-validated backend comparison (more robust than single split)."""
    print("=" * 80)
    print(f"CROSS-VALIDATED BACKEND COMPARISON ({n_folds}-fold GroupKFold)")
    print("=" * 80)
    print()

    import iterations.v4_hgb_momentum.train  # noqa: F401
    import iterations.v1_hgb_baseline.train as v1

    print("Loading data...")
    df = v1.build_weekly_snapshot_df(zip_path, json_name)
    print(f"  {len(df)} rows loaded")
    print()

    backends = _get_available_backends()
    results = []

    for pos in POSITIONS:
        pos_df = df[df["position"] == pos].copy()
        if len(pos_df) < 100:
            continue

        pos_features = v1.get_features_for_position(pos)
        X = pos_df[pos_features].values
        y = pos_df["log_ratio"].values
        start_ktc = pos_df["start_ktc"].values
        end_ktc = pos_df["end_ktc"].values
        groups = pos_df["player_id"].values

        n_unique = len(np.unique(groups))
        actual_folds = min(n_folds, n_unique)
        gkf = GroupKFold(n_splits=actual_folds)

        print(f"[{pos}] {len(pos_df)} samples, {n_unique} players, {actual_folds} folds")

        imputer = SimpleImputer(strategy="median")

        for name, builder in backends.items():
            fold_maes = []
            for fold, (train_idx, test_idx) in enumerate(gkf.split(X, y, groups)):
                try:
                    X_tr, X_te = X[train_idx], X[test_idx]
                    y_tr = y[train_idx]
                    sktc_te = start_ktc[test_idx]
                    ektc_te = end_ktc[test_idx]

                    model = builder(pos)

                    if name in ("XGBoost",):
                        X_tr_imp = imputer.fit_transform(X_tr)
                        X_te_imp = imputer.transform(X_te)
                        model.fit(X_tr_imp, y_tr)
                        preds = model.predict(X_te_imp)
                    elif name == "CatBoost":
                        model.fit(X_tr, y_tr, verbose=False)
                        preds = model.predict(X_te)
                    else:
                        model.fit(X_tr, y_tr)
                        preds = model.predict(X_te)

                    pred_end = sktc_te * np.exp(np.clip(preds, -1.5, 2.0))
                    fold_mae = mean_absolute_error(ektc_te, pred_end)
                    fold_maes.append(fold_mae)
                except Exception as e:
                    print(f"    {name} fold {fold} failed: {e}")

            if fold_maes:
                mean_mae = np.mean(fold_maes)
                std_mae = np.std(fold_maes)
                results.append({"position": pos, "backend": name,
                                "mean_mae": mean_mae, "std_mae": std_mae})
                print(f"  {name:<12} MAE={mean_mae:>7.1f} ± {std_mae:.1f}")

        # Baseline
        bl_maes = []
        for _, test_idx in gkf.split(X, y, groups):
            bl_mae = mean_absolute_error(end_ktc[test_idx], start_ktc[test_idx])
            bl_maes.append(bl_mae)
        bl_mean = np.mean(bl_maes)
        print(f"  {'Baseline':<12} MAE={bl_mean:>7.1f}")
        print()

    if results:
        print("\nSUMMARY (CV Mean MAE):")
        rdf = pd.DataFrame(results)
        pivot = rdf.pivot_table(index="position", columns="backend", values="mean_mae", aggfunc="first")
        print(pivot.round(1).to_string())
        print()


# ── Main ───────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Evaluate KTC prediction model")
    parser.add_argument("--model-dir", default="iterations/v4_hgb_momentum/models",
                        help="Path to model directory with test_predictions.csv")
    parser.add_argument("--compare-backends", action="store_true",
                        help="Compare HGB vs XGBoost vs LightGBM vs CatBoost")
    parser.add_argument("--cv", action="store_true",
                        help="Use cross-validation for backend comparison (slower, more robust)")
    parser.add_argument("--zip", default="data/training-data.zip",
                        help="Path to training data zip")
    parser.add_argument("--json", default="training-data.json")
    parser.add_argument("--worst", type=int, default=25,
                        help="Number of worst misses to show")
    args = parser.parse_args()

    # Load test predictions
    df = load_test_predictions(args.model_dir)
    print(f"Loaded {len(df)} test predictions from {args.model_dir}")
    print(f"Positions: {df.groupby('position').size().to_dict()}")
    print(f"Years: {sorted(df['year'].unique())}")
    print()

    # Run evaluation reports
    print_baseline_comparison(df)
    print_mae_by_bucket(df)
    print_mae_by_age(df)
    print_worst_misses(df, n=args.worst)
    print_best_predictions(df)

    # Optional backend comparison
    if args.compare_backends:
        if args.cv:
            compare_backends_cv(args.zip, args.json)
        else:
            compare_backends(args.zip, args.json)


if __name__ == "__main__":
    main()
