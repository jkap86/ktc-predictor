"""Optuna hyperparameter tuning for v4 HGB models.

Searches over loss function, tree depth, learning rate, iterations,
min_samples_leaf, and l2_regularization per position.

Uses GroupKFold CV with end_ktc MAE as the objective (same metric we report).

Usage:
    cd backend
    python -m iterations.v4_hgb_momentum.tune --zip data/training-data.zip
    python -m iterations.v4_hgb_momentum.tune --zip data/training-data.zip --positions QB WR --n-trials 80
"""

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import optuna
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.metrics import mean_absolute_error
from sklearn.model_selection import GroupKFold, GroupShuffleSplit

# Import v4's patched pipeline (applies v3 → v1 monkey-patches)
import iterations.v4_hgb_momentum.train as v4  # noqa: F401 — triggers patches
import iterations.v1_hgb_baseline.train as v1

optuna.logging.set_verbosity(optuna.logging.WARNING)

POSITIONS = ["QB", "RB", "WR", "TE"]
N_CV_FOLDS = 5


def _objective(trial, X, y, start_ktc, groups, position):
    """Optuna objective: GroupKFold CV mean end_ktc MAE."""
    loss = trial.suggest_categorical("loss", ["squared_error", "absolute_error"])
    max_depth = trial.suggest_int("max_depth", 3, 8)
    learning_rate = trial.suggest_float("learning_rate", 0.01, 0.15, log=True)
    max_iter = trial.suggest_int("max_iter", 100, 600, step=50)
    min_samples_leaf = trial.suggest_int("min_samples_leaf", 5, 50)
    l2_regularization = trial.suggest_float("l2_regularization", 1e-3, 10.0, log=True)
    max_leaf_nodes = trial.suggest_int("max_leaf_nodes", 20, 80, step=5)

    monotonic_cst = v1._build_monotonic_constraints(position)

    gkf = GroupKFold(n_splits=N_CV_FOLDS)
    fold_maes = []

    for train_idx, val_idx in gkf.split(X, y, groups):
        model = HistGradientBoostingRegressor(
            loss=loss,
            max_iter=max_iter,
            max_depth=max_depth,
            learning_rate=learning_rate,
            min_samples_leaf=min_samples_leaf,
            l2_regularization=l2_regularization,
            max_leaf_nodes=max_leaf_nodes,
            monotonic_cst=monotonic_cst,
            random_state=42,
        )
        model.fit(X[train_idx], y[train_idx])
        preds_log = model.predict(X[val_idx])

        # Reconstruct end_ktc for MAE (the metric we actually care about)
        pred_end_ktc = start_ktc[val_idx] * np.exp(preds_log)
        actual_end_ktc = start_ktc[val_idx] * np.exp(y[val_idx])
        fold_maes.append(mean_absolute_error(actual_end_ktc, pred_end_ktc))

    return np.mean(fold_maes)


def tune_position(pos, df, n_trials=60, seed=42):
    """Run Optuna study for a single position."""
    pos_df = df[df["position"] == pos].copy()
    pos_features = v1.get_features_for_position(pos)

    X = pos_df[pos_features].values
    y = pos_df["log_ratio"].values
    start_ktc = pos_df["start_ktc"].values
    groups = pos_df["player_id"].values

    # Hold out 20% for final eval (same split as training)
    gss = GroupShuffleSplit(n_splits=1, test_size=0.2, random_state=seed)
    train_idx, test_idx = next(gss.split(X, y, groups))

    X_train = X[train_idx]
    y_train = y[train_idx]
    start_ktc_train = start_ktc[train_idx]
    groups_train = groups[train_idx]

    # Holdout for final comparison
    X_test = X[test_idx]
    y_test = y[test_idx]
    start_ktc_test = start_ktc[test_idx]

    print(f"\n{'='*60}")
    print(f"[{pos}] Tuning {n_trials} trials on {len(X_train)} samples...")
    print(f"  Features: {len(pos_features)}")
    print(f"  Holdout: {len(X_test)} samples")

    study = optuna.create_study(
        direction="minimize",
        sampler=optuna.samplers.TPESampler(seed=seed),
    )
    study.optimize(
        lambda trial: _objective(
            trial, X_train, y_train, start_ktc_train, groups_train, pos
        ),
        n_trials=n_trials,
        show_progress_bar=True,
    )

    best = study.best_params
    best_cv_mae = study.best_value

    # Compare with current params on the same holdout
    current_params = v1.POSITION_HYPERPARAMS.get(pos, v1.POSITION_HYPERPARAMS["WR"])
    monotonic_cst = v1._build_monotonic_constraints(pos)

    # Current model on holdout
    current_model = HistGradientBoostingRegressor(
        max_iter=current_params["n_estimators"],
        max_depth=current_params["max_depth"],
        learning_rate=current_params["learning_rate"],
        monotonic_cst=monotonic_cst,
        random_state=42,
    )
    current_model.fit(X_train, y_train)
    current_preds = current_model.predict(X_test)
    current_mae = mean_absolute_error(
        start_ktc_test * np.exp(y_test),
        start_ktc_test * np.exp(current_preds),
    )

    # Tuned model on holdout
    tuned_model = HistGradientBoostingRegressor(
        loss=best["loss"],
        max_iter=best["max_iter"],
        max_depth=best["max_depth"],
        learning_rate=best["learning_rate"],
        min_samples_leaf=best["min_samples_leaf"],
        l2_regularization=best["l2_regularization"],
        max_leaf_nodes=best["max_leaf_nodes"],
        monotonic_cst=monotonic_cst,
        random_state=42,
    )
    tuned_model.fit(X_train, y_train)
    tuned_preds = tuned_model.predict(X_test)
    tuned_mae = mean_absolute_error(
        start_ktc_test * np.exp(y_test),
        start_ktc_test * np.exp(tuned_preds),
    )

    delta = tuned_mae - current_mae
    pct = delta / current_mae * 100

    print(f"\n  Best CV MAE: {best_cv_mae:.1f}")
    print(f"  Current holdout MAE: {current_mae:.1f}")
    print(f"  Tuned holdout MAE:   {tuned_mae:.1f}  ({pct:+.1f}%)")
    print(f"  Best params: {json.dumps(best, indent=4)}")

    return {
        "position": pos,
        "best_params": best,
        "best_cv_mae": round(best_cv_mae, 1),
        "current_holdout_mae": round(current_mae, 1),
        "tuned_holdout_mae": round(tuned_mae, 1),
        "delta_mae": round(delta, 1),
        "delta_pct": round(pct, 1),
        "n_trials": n_trials,
        "n_train": len(X_train),
        "n_test": len(X_test),
    }


def main():
    parser = argparse.ArgumentParser(description="Optuna hyperparameter tuning for v4")
    parser.add_argument("--zip", default="data/training-data.zip")
    parser.add_argument("--json", default="training-data.json")
    parser.add_argument("--n-trials", type=int, default=60)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--positions", nargs="+", default=POSITIONS)
    parser.add_argument("--out", default="iterations/v4_hgb_momentum/tuned_params.json")
    args = parser.parse_args()

    print("Loading data...")
    df = v1.build_weekly_snapshot_df(args.zip, args.json)
    print(f"  Total rows: {len(df)}")

    results = {}
    for pos in args.positions:
        if pos not in POSITIONS:
            print(f"  Skipping unknown position: {pos}")
            continue
        result = tune_position(pos, df, n_trials=args.n_trials, seed=args.seed)
        results[pos] = result

    # Save results
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nResults saved to {out_path}")

    # Summary table
    print(f"\n{'='*60}")
    print("SUMMARY")
    print(f"{'='*60}")
    print(f"{'Pos':<5} {'Current':>10} {'Tuned':>10} {'Delta':>10} {'Loss':<16}")
    for pos, r in results.items():
        print(
            f"{pos:<5} {r['current_holdout_mae']:>10.1f} {r['tuned_holdout_mae']:>10.1f} "
            f"{r['delta_mae']:>+10.1f} {r['best_params']['loss']:<16}"
        )


if __name__ == "__main__":
    main()
