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
from sklearn.linear_model import Ridge, ElasticNet
from sklearn.metrics import mean_absolute_error, r2_score
from sklearn.model_selection import GridSearchCV, GroupKFold, GroupShuffleSplit
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import FunctionTransformer, PolynomialFeatures, StandardScaler

from .calibration import MonotoneLinearCalibrator
from .data import build_weekly_snapshot_df
from .io import EnsembleModel, save_bundle
from .knn_adjustment import EliteKNNAdjuster

POSITIONS = ["QB", "RB", "WR", "TE"]
MODEL_TYPES = ["hgb", "ridge", "elasticnet-poly"]

# Feature flag for offseason features (disabled - hurt model performance)
# Set to True only when testing with tree-based models (HGB, XGBoost, etc.)
USE_OFFSEASON_FEATURES = False

# Feature flag for KNN elite adjustment (reduces under-prediction of elite risers)
USE_KNN_ADJUSTMENT = True
KNN_ELITE_THRESHOLD = 6000
KNN_K = 10
KNN_BLEND_WEIGHT = 0.3

# Feature flag for contract features (disabled by default until validated)
USE_CONTRACT_FEATURES = True

# Feature flag for PFF grades (captures efficiency beyond PPG)
# Disabled: sparse coverage (55% for elite RB) adds noise
USE_PFF_FEATURES = False

# Feature flag for target variable
# False: use log_ratio = log(end_ktc / start_ktc) [current]
# True: use pct_change = (end_ktc - start_ktc) / start_ktc [experiment]
# pct_change treats all tiers equally; log_ratio compresses elite tier errors
USE_PCT_CHANGE_TARGET = False  # Experiment B failed - extreme outliers destabilize training

# ============================================================================
# FEATURE LISTS FOR MIXED PROCESSING PIPELINE
# ============================================================================
# Core features go through polynomial expansion (interactions matter)
# Linear features bypass polynomial to avoid noise amplification
#
# POSITION-SPECIFIC FEATURES:
# Prior-year KTC features (ktc_yoy_log, ktc_peak_drawdown, has_prior_season)
# are QB-ONLY. They helped QB (MAE 799→701) but hurt RB 6k+ tier badly.
# This is because QB trajectory is a stable latent factor, while RB elite
# tier is dominated by variance (injuries, role shocks).

# Core features: polynomial expansion enabled
# These benefit from interaction terms (e.g., ppg * games_played)
_CORE_FEATURES = [
    "games_played_so_far",
    "ppg_so_far",
    "weeks_missed_so_far",
    "draft_pick",
    "years_remaining",
    "start_ktc_quartile",
    "age_prime_distance",
    "is_breakout_candidate",
]

# Base linear features (all positions): bypass polynomial expansion
_BASE_LINEAR_FEATURES = [
    "start_ktc",
    "start_ktc_was_sentinel",
]

# Prior-season KTC features (QB ONLY - captures trajectory/path)
# These help QB predict upside but destabilize RB elite tier predictions.
_PRIOR_SEASON_FEATURES = [
    "ktc_yoy_log",         # log(start_ktc / prior_end_ktc), clipped
    "ktc_peak_drawdown",   # log(start_ktc / max_ktc_prior)
    "has_prior_season",    # 1 if prior season data exists
]

# Prior-season PPG features (QB ONLY - captures performance trajectory)
# Complements KTC trajectory by tracking actual on-field performance.
_PRIOR_PPG_FEATURES = [
    "prior_ppg",           # Prior season's PPG (absolute baseline)
    "ppg_yoy_log",         # log(ppg_so_far / prior_ppg), clipped to [-1.0, 1.0]
    "has_prior_ppg",       # 1 if valid prior PPG exists
]

# Contract features (from nfl_data_py)
# These help distinguish rookie deals vs established vets, contract year performers, etc.
_CONTRACT_FEATURES = [
    "apy_cap_pct",         # APY as % of salary cap (0.0 to ~0.25)
    "is_contract_year",    # 1 if in final year, 0 otherwise
    "apy_position_rank",   # Percentile of APY within position (0-1)
    "has_contract_data",   # 1 if contract info exists, else 0
]

# PFF grades (efficiency beyond PPG)
# Particularly helpful for RB where raw PPG doesn't capture quality of touches.
_PFF_FEATURES = [
    "pff_overall_grade",   # 0-100 composite efficiency rating
    "pff_grade_tier",      # 0=reserve, 1=starter, 2=elite
    "pff_position_grade",  # Position-specific grade (run/receiving)
    "has_pff_data",        # 1 if PFF data exists, else 0
]

# Team context features
# Captures opportunity quality and positional competition
_TEAM_FEATURES = [
    "qb_ktc",                  # QB KTC value (passing game quality for RB/WR)
    "team_total_ktc",          # Sum of teammates' KTC (roster strength)
    "positional_competition",  # KTC of same-position teammates (committee risk)
]

# Feature flag for team features
# Disabled: adds noise to RB 6k+ tier predictions
USE_TEAM_FEATURES = False

# Offseason features (only used when USE_OFFSEASON_FEATURES=True)
# These hurt ElasticNet+Poly (RB MAE 558→602) due to polynomial interactions,
# but may help tree-based models that can isolate conditional interactions.
_OFFSEASON_FEATURES = [
    "offseason_percentile",  # 0=at low, 1=at high (mean reversion)
    "trend_14d",  # log(last/14d_ago) - late offseason momentum
]


def get_features_for_position(position: str) -> list[str]:
    """Get the feature list for a specific position.

    QB, WR, TE use prior-season KTC features (trajectory signal is stable).
    These positions also use prior-season PPG features (performance trajectory).
    RB does NOT use these features (variance dominates at elite tier - injuries, role shocks).

    All positions use contract features when USE_CONTRACT_FEATURES=True.
    """
    linear_features = _BASE_LINEAR_FEATURES.copy()

    # Prior-season trajectory features (KTC + PPG) for stable positions
    # RB excluded due to high variance at elite tier (features hurt more than help)
    if position in ("QB", "WR", "TE"):
        linear_features.extend(_PRIOR_SEASON_FEATURES)
        linear_features.extend(_PRIOR_PPG_FEATURES)

    # Contract features for all positions (market position, contract year effect)
    if USE_CONTRACT_FEATURES:
        linear_features.extend(_CONTRACT_FEATURES)

    # PFF grades for all positions (efficiency beyond PPG)
    if USE_PFF_FEATURES:
        linear_features.extend(_PFF_FEATURES)

    # Team context features (opportunity quality, competition)
    if USE_TEAM_FEATURES:
        linear_features.extend(_TEAM_FEATURES)

    if USE_OFFSEASON_FEATURES:
        return _CORE_FEATURES + _OFFSEASON_FEATURES + linear_features
    else:
        return _CORE_FEATURES + linear_features


def get_feature_counts_for_position(position: str) -> tuple[int, int]:
    """Get (n_core_features, n_linear_features) for a position."""
    features = get_features_for_position(position)
    n_core = len(_CORE_FEATURES)
    n_linear = len(features) - n_core
    return n_core, n_linear


# Default FEATURES list (used for non-position-specific contexts)
# This is the QB feature set (superset) for backwards compatibility
FEATURES = get_features_for_position("QB")
N_CORE_FEATURES = len(_CORE_FEATURES)
N_LINEAR_FEATURES = len(FEATURES) - N_CORE_FEATURES
MIN_SAMPLES = 100
GP_BUCKETS = [(1, 3), (4, 7), (8, 11), (12, 17)]
GP_BUCKET_MIN_SAMPLES = 30

# Ensemble configuration: train multiple models with different seeds for variance reduction
ENSEMBLE_SEEDS = [42, 123, 456, 789, 999]

# Position-specific hyperparameters: QB/RB/TE underfit with default params
# WR works great with defaults (R²=0.91), so keep current settings
POSITION_HYPERPARAMS = {
    "QB": {"max_depth": 6, "learning_rate": 0.08, "n_estimators": 300},
    "RB": {"max_depth": 7, "learning_rate": 0.05, "n_estimators": 400},
    "WR": {"max_depth": 5, "learning_rate": 0.10, "n_estimators": 200},
    "TE": {"max_depth": 6, "learning_rate": 0.08, "n_estimators": 300},
}

NAN_CHECK_COLS = ["age", "draft_pick", "years_remaining"]

# ElasticNet hyperparameter grid for cross-validated tuning
ELASTICNET_PARAM_GRID = {
    'enet__alpha': [0.001, 0.005, 0.01, 0.05, 0.1],
    'enet__l1_ratio': [0.3, 0.5, 0.7, 0.9],
}


def _gp_bucket_key(gp: float) -> str | None:
    """Return the bucket key string for a games_played value, or None."""
    for lo, hi in GP_BUCKETS:
        if lo <= gp <= hi:
            return f"gp_{lo}_{hi}"
    return None


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


def compute_bias_diagnostics(test_rows: list[dict]) -> dict:
    """Compute detailed bias diagnostics for model monitoring.

    Returns a dict with bias/MAE by position, tier, and risers/fallers.
    This is saved as diagnostics.json for regression testing.
    """
    if not test_rows:
        return {}

    test_df = pd.DataFrame(test_rows)
    test_df["residual"] = test_df["actual_end_ktc"] - test_df["predicted_end_ktc"]

    diagnostics = {"positions": {}}

    for pos in POSITIONS:
        pos_df = test_df[test_df["position"] == pos]
        if pos_df.empty:
            continue

        pos_diag = {
            "n": len(pos_df),
            "mae": round(pos_df["residual"].abs().mean(), 1),
            "bias": round(pos_df["residual"].mean(), 1),
            "tiers": {},
        }

        # Bias by KTC tier
        tier_bounds = [(0, 2000, "0-2k"), (2000, 4000, "2k-4k"), (4000, 6000, "4k-6k"), (6000, float("inf"), "6k+")]
        for lo, hi, name in tier_bounds:
            tier_df = pos_df[(pos_df["start_ktc"] >= lo) & (pos_df["start_ktc"] < hi)]
            if len(tier_df) >= 20:
                tier_info = {
                    "n": len(tier_df),
                    "mae": round(tier_df["residual"].abs().mean(), 1),
                    "bias": round(tier_df["residual"].mean(), 1),
                }
                # Risers vs fallers for high tiers
                if lo >= 4000:
                    risers = tier_df[tier_df["actual_end_ktc"] > tier_df["start_ktc"]]
                    fallers = tier_df[tier_df["actual_end_ktc"] <= tier_df["start_ktc"]]
                    if len(risers) >= 10:
                        tier_info["riser_bias"] = round(risers["residual"].mean(), 1)
                        tier_info["riser_n"] = len(risers)
                    if len(fallers) >= 10:
                        tier_info["faller_bias"] = round(fallers["residual"].mean(), 1)
                        tier_info["faller_n"] = len(fallers)
                pos_diag["tiers"][name] = tier_info

        diagnostics["positions"][pos] = pos_diag

    return diagnostics


def _try_xgb(seed: int, position: str = "WR"):
    """Attempt to create an XGBRegressor with monotonic constraints and position-specific hyperparams."""
    try:
        from xgboost import XGBRegressor

        params = POSITION_HYPERPARAMS.get(position, POSITION_HYPERPARAMS["WR"])
        # Monotonic constraints: 12 features
        # ppg (idx 1): positive, ppg_zscore (idx 10): positive,
        # is_breakout_candidate (idx 11): positive
        return XGBRegressor(
            n_estimators=params["n_estimators"],
            max_depth=params["max_depth"],
            learning_rate=params["learning_rate"],
            monotone_constraints="(0,1,0,0,0,0,0,0,0,0,1,1)",
            random_state=seed,
        )
    except ImportError:
        return None


def _build_monotonic_constraints(position: str) -> list[int]:
    """Build monotonic constraints matching get_features_for_position() layout.

    Constraint values: 0 = no constraint, 1 = positive monotonic, -1 = negative monotonic.
    Must stay in sync with get_features_for_position() feature order.
    """
    # Core features constraints (8 features)
    # ppg_so_far (idx 1): positive monotonic - higher PPG should predict higher KTC
    # is_breakout_candidate (idx 7): positive monotonic - breakout candidates should predict higher KTC
    core_constraints = [0, 1, 0, 0, 0, 0, 0, 1]

    # Base linear features constraints (2 features)
    # start_ktc: positive monotonic - higher start should predict higher end
    # start_ktc_was_sentinel: no constraint
    linear_constraints = [1, 0]

    # Prior-season features for stable trajectory positions (QB, WR, TE)
    # RB excluded due to high variance at elite tier
    if position in ("QB", "WR", "TE"):
        # Prior KTC features (3): ktc_yoy_log, ktc_peak_drawdown, has_prior_season
        linear_constraints.extend([1, 0, 0])
        # Prior PPG features (3): prior_ppg, ppg_yoy_log, has_prior_ppg
        linear_constraints.extend([0, 1, 0])

    # Contract features (4): apy_cap_pct, is_contract_year, apy_position_rank, has_contract_data
    if USE_CONTRACT_FEATURES:
        linear_constraints.extend([0, 0, 0, 0])

    # PFF features (4): pff_overall_grade, pff_grade_tier, pff_position_grade, has_pff_data
    if USE_PFF_FEATURES:
        linear_constraints.extend([0, 0, 0, 0])

    # Team features (3): qb_ktc, team_total_ktc, positional_competition
    if USE_TEAM_FEATURES:
        linear_constraints.extend([0, 0, 0])

    return core_constraints + linear_constraints


def _build_hgb(seed: int, position: str = "WR"):
    """Create a HistGradientBoostingRegressor with monotonic constraints and position-specific hyperparams."""
    params = POSITION_HYPERPARAMS.get(position, POSITION_HYPERPARAMS["WR"])
    monotonic_cst = _build_monotonic_constraints(position)
    return HistGradientBoostingRegressor(
        max_iter=params["n_estimators"],
        max_depth=params["max_depth"],
        learning_rate=params["learning_rate"],
        monotonic_cst=monotonic_cst,
        random_state=seed,
    )


def _build_quantile_hgb(seed: int, position: str, quantile: float):
    """Create a HistGradientBoostingRegressor for quantile regression.

    Used for uncertainty estimation - trains models to predict 20th and 80th percentiles.
    """
    params = POSITION_HYPERPARAMS.get(position, POSITION_HYPERPARAMS["WR"])
    monotonic_cst = _build_monotonic_constraints(position)
    return HistGradientBoostingRegressor(
        loss="quantile",
        quantile=quantile,
        max_iter=params["n_estimators"],
        max_depth=params["max_depth"],
        learning_rate=params["learning_rate"],
        monotonic_cst=monotonic_cst,
        random_state=seed,
    )


def _build_ridge():
    """Create a Ridge regression model with imputation and standardization.

    Simple linear model with L2 regularization.
    Uses median imputation for NaN values (age, draft_pick, years_remaining).
    """
    return Pipeline([
        ('imputer', SimpleImputer(strategy='median')),
        ('scaler', StandardScaler()),
        ('ridge', Ridge(alpha=1.0))
    ])


def _build_elasticnet_poly(
    alpha: float = 0.01,
    l1_ratio: float = 0.5,
    n_core_features: int | None = None,
    n_total_features: int | None = None,
):
    """Create an ElasticNet model with mixed feature processing.

    Uses ColumnTransformer to split feature processing:
    - Core features (0 to n_core_features-1): polynomial expansion for interactions
    - Linear features (n_core_features to end): linear-only passthrough

    This prevents noise amplification from polynomial expansion on sensitive
    features like ktc_yoy_log and ktc_peak_drawdown.

    Parameters
    ----------
    alpha : float
        Regularization strength (higher = more regularization).
    l1_ratio : float
        Balance between L1 and L2 (1.0 = pure L1/Lasso, 0.0 = pure L2/Ridge).
    n_core_features : int or None
        Number of core features for polynomial expansion. Defaults to N_CORE_FEATURES.
    n_total_features : int or None
        Total number of features. Defaults to len(FEATURES).
    """
    # Use defaults if not specified (for backwards compatibility)
    if n_core_features is None:
        n_core_features = N_CORE_FEATURES
    if n_total_features is None:
        n_total_features = len(FEATURES)

    # Core features: impute → scale → polynomial interactions
    core_pipeline = Pipeline([
        ('imputer', SimpleImputer(strategy='median')),
        ('scaler', StandardScaler()),
        ('poly', PolynomialFeatures(degree=2, interaction_only=True, include_bias=False))
    ])

    # Linear features: impute → scale (no polynomial)
    linear_pipeline = Pipeline([
        ('imputer', SimpleImputer(strategy='median')),
        ('scaler', StandardScaler())
    ])

    # Column indices for each feature group
    core_indices = list(range(n_core_features))
    linear_indices = list(range(n_core_features, n_total_features))

    # Combine both feature groups
    preprocessor = ColumnTransformer([
        ('core', core_pipeline, core_indices),
        ('linear', linear_pipeline, linear_indices)
    ])

    return Pipeline([
        ('preprocessor', preprocessor),
        ('enet', ElasticNet(alpha=alpha, l1_ratio=l1_ratio, max_iter=10000))
    ])


def _build_elasticnet_poly_tuned(
    X_train, y_train, groups, start_ktc_train,
    n_cv_folds: int = 5,
    n_core_features: int | None = None,
    n_total_features: int | None = None,
):
    """Build ElasticNet+Poly with cross-validated hyperparameter search.

    Uses GroupKFold to prevent player leakage during CV.
    Scores on end_ktc MAE (not log_ratio MAE) to align with evaluation metric.

    Parameters
    ----------
    X_train : np.ndarray
        Training features.
    y_train : np.ndarray
        Training targets (log_ratio).
    groups : np.ndarray
        Player IDs for group-based CV splitting.
    start_ktc_train : np.ndarray
        Start KTC values for reconstructing end_ktc in scorer.
    n_cv_folds : int
        Number of cross-validation folds.
    n_core_features : int or None
        Number of core features for polynomial expansion. Defaults to N_CORE_FEATURES.
    n_total_features : int or None
        Total number of features. Defaults to len(FEATURES).

    Returns
    -------
    tuple
        (best_estimator, best_params) - fitted Pipeline and dict of best hyperparameters.
    """
    # Use defaults if not specified (for backwards compatibility)
    if n_core_features is None:
        n_core_features = N_CORE_FEATURES
    if n_total_features is None:
        n_total_features = len(FEATURES)

    # Ensure we have enough unique groups for CV
    n_unique_groups = len(np.unique(groups))
    actual_folds = min(n_cv_folds, n_unique_groups)

    if actual_folds < 2:
        print(f"  Warning: Only {n_unique_groups} unique groups, using default params")
        model = _build_elasticnet_poly(
            n_core_features=n_core_features, n_total_features=n_total_features
        )
        model.fit(X_train, y_train)
        return model, {'enet__alpha': 0.01, 'enet__l1_ratio': 0.5}

    # Append start_ktc as the last column of X for the scorer to access
    # The pipeline will strip it before training via FunctionTransformer
    X_with_ktc = np.column_stack([X_train, start_ktc_train])

    # Build ColumnTransformer for mixed feature processing
    # Note: indices are for X_train (without the appended start_ktc column)
    core_pipeline = Pipeline([
        ('imputer', SimpleImputer(strategy='median')),
        ('scaler', StandardScaler()),
        ('poly', PolynomialFeatures(degree=2, interaction_only=True, include_bias=False))
    ])
    linear_pipeline = Pipeline([
        ('imputer', SimpleImputer(strategy='median')),
        ('scaler', StandardScaler())
    ])
    core_indices = list(range(n_core_features))
    linear_indices = list(range(n_core_features, n_total_features))

    preprocessor = ColumnTransformer([
        ('core', core_pipeline, core_indices),
        ('linear', linear_pipeline, linear_indices)
    ])

    # Pipeline that drops the last column (start_ktc) before processing
    base_pipeline = Pipeline([
        ('drop_ktc', FunctionTransformer(lambda X: X[:, :-1], validate=False)),
        ('preprocessor', preprocessor),
        ('enet', ElasticNet(max_iter=10000))
    ])

    def end_ktc_mae_scorer(estimator, X, y_log_ratio):
        """Custom scorer: compute MAE on reconstructed end_ktc.

        This aligns CV scoring with the actual evaluation metric.
        """
        start_ktc = X[:, -1]  # Last column is start_ktc
        y_pred_log = estimator.predict(X)

        pred_end_ktc = start_ktc * np.exp(y_pred_log)
        actual_end_ktc = start_ktc * np.exp(y_log_ratio)

        # Return negative because sklearn maximizes scores
        return -mean_absolute_error(actual_end_ktc, pred_end_ktc)

    gkf = GroupKFold(n_splits=actual_folds)

    grid_search = GridSearchCV(
        base_pipeline,
        ELASTICNET_PARAM_GRID,
        cv=gkf,
        scoring=end_ktc_mae_scorer,
        n_jobs=-1,
        verbose=1
    )

    grid_search.fit(X_with_ktc, y_train, groups=groups)

    best_alpha = grid_search.best_params_['enet__alpha']
    best_l1 = grid_search.best_params_['enet__l1_ratio']

    print(f"  Best params: {grid_search.best_params_}")
    print(f"  Best CV MAE (end_ktc): {-grid_search.best_score_:.1f}")

    # Return a clean model (without the ktc-dropping step) retrained on full data
    final_model = _build_elasticnet_poly(
        alpha=best_alpha, l1_ratio=best_l1,
        n_core_features=n_core_features, n_total_features=n_total_features
    )
    final_model.fit(X_train, y_train)

    return final_model, grid_search.best_params_


def _monotonic_smoke_test(model, position: str) -> bool:
    """Verify log_ratio predictions are non-decreasing as PPG increases (gp=8, start_ktc=5000)."""
    ppg_values = [5, 10, 15, 20]

    # Build feature vectors based on position
    # Core features (8): gp, ppg, weeks_missed, draft_pick, years_remaining, ktc_quartile, age_prime_dist, is_breakout
    # Base linear (2): start_ktc, sentinel
    # For stable positions (QB, WR, TE) at age 25, start_ktc 5000 (Q4)
    X_test = []
    for ppg in ppg_values:
        # Core features
        row = [8, ppg, 2, np.nan, 3, 4, -2, 0]  # gp, ppg, weeks_missed, draft_pick, years_remaining, ktc_quartile, age_prime_dist, is_breakout
        # Base linear features
        row.extend([5000, 0])  # start_ktc, start_ktc_was_sentinel

        # Prior-season features for stable trajectory positions
        if position in ("QB", "WR", "TE"):
            # Prior season KTC features (3): ktc_yoy_log, ktc_peak_drawdown, has_prior_season
            row.extend([0.0, 0.0, 1])
            # Prior PPG features (3): prior_ppg, ppg_yoy_log, has_prior_ppg
            row.extend([15.0, np.log((ppg + 0.1) / (15 + 0.1)), 1])

        # Contract features (4): apy_cap_pct, is_contract_year, apy_position_rank, has_contract_data
        if USE_CONTRACT_FEATURES:
            row.extend([0.05, 0, 0.5, 1])

        # PFF features (4): pff_overall_grade, pff_grade_tier, pff_position_grade, has_pff_data
        if USE_PFF_FEATURES:
            row.extend([70.0, 1, 70.0, 1])

        # Team features (3): qb_ktc, team_total_ktc, positional_competition
        if USE_TEAM_FEATURES:
            row.extend([5000, 30000, 3000])

        X_test.append(row)

    X_test = np.array(X_test)
    preds = model.predict(X_test)

    is_monotonic = all(preds[i] <= preds[i + 1] for i in range(len(preds) - 1))
    status = "PASS" if is_monotonic else "FAIL"
    print(f"  Monotonic test ({position}): {status}  preds={[round(p, 4) for p in preds]}")
    return is_monotonic


def _ppg_sensitivity_test(model, calibrator_dict, clip_bounds, position: str) -> bool:
    """Verify predictions vary meaningfully across PPG range (not collapsed by calibration)."""
    from .predict import predict_end_ktc

    ppg_values = [0, 10, 20, 30]
    results = []
    for ppg in ppg_values:
        r = predict_end_ktc(
            models={position: model},
            clip_bounds={position: clip_bounds},
            calibrators={position: calibrator_dict},
            position=position,
            gp=17,
            ppg=ppg,
            start_ktc=5000,
        )
        results.append(r["end_ktc"])

    # Require std dev > 100 (meaningful spread across PPG range)
    std_dev = float(np.std(results))
    is_sensitive = std_dev > 100

    status = "PASS" if is_sensitive else "FAIL"
    print(f"  PPG sensitivity ({position}): {status}  std={std_dev:.1f}  preds={[round(r, 0) for r in results]}")
    return is_sensitive


def train_all(
    zip_path: str,
    json_name: str = "training-data.json",
    out_dir: str = "models",
    test_size: float = 0.2,
    seed: int = 42,
    no_calibration: bool = False,
    prefer_xgb: bool = False,
    export_csv: str | None = None,
    model_type: str = "hgb",
    tune_hyperparams: bool = False,
) -> dict:
    """Train per-position models and return bundle.

    Parameters
    ----------
    model_type : str
        One of "hgb" (HistGradientBoosting), "ridge", or "elasticnet-poly".
    tune_hyperparams : bool
        If True and model_type is "elasticnet-poly", run GridSearchCV to find
        optimal alpha and l1_ratio per position.
    """
    if model_type not in MODEL_TYPES:
        raise ValueError(f"model_type must be one of {MODEL_TYPES}, got '{model_type}'")

    is_linear = model_type in ("ridge", "elasticnet-poly")

    print(f"Loading data from {zip_path}...")
    print(f"Model type: {model_type}" + (" (with tuning)" if tune_hyperparams else ""))
    df = build_weekly_snapshot_df(zip_path, json_name)
    print(f"  Total rows: {len(df)}")
    print(f"  Positions: {df.groupby('position').size().to_dict()}")
    print()

    bundle = {
        "models": {},
        "clip_bounds": {},
        "calibrators": {},
        "quantile_models": {},
        "metrics": {},
        # Position-specific feature contracts (QB has extra prior-season features)
        "feature_names": {pos: get_features_for_position(pos) for pos in POSITIONS},
    }
    all_test_rows: list[dict] = []

    for pos in POSITIONS:
        pos_df = df[df["position"] == pos].copy()
        n = len(pos_df)

        if n < MIN_SAMPLES:
            print(f"[{pos}] Skipping: only {n} samples (need >= {MIN_SAMPLES})")
            print()
            continue

        print(f"[{pos}] Training on {n} samples...")

        # Get position-specific features (QB uses prior-season KTC features)
        pos_features = get_features_for_position(pos)
        n_core, n_linear = get_feature_counts_for_position(pos)

        X = pos_df[pos_features].values
        # Target variable: log_ratio or pct_change based on flag
        if USE_PCT_CHANGE_TARGET:
            y = pos_df["pct_change"].values
            target_name = "pct_change"
        else:
            y = pos_df["log_ratio"].values
            target_name = "log_ratio"
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

        # Select backend and build model(s)
        backend_name = model_type.upper()
        quantile_models = {}

        # Track best hyperparameters for saving to metrics
        best_params = None

        if model_type == "ridge":
            model = _build_ridge()
            model.fit(X_train, y_train)
            print(f"  Backend: Ridge regression (linear)")
        elif model_type == "elasticnet-poly":
            train_groups = groups[train_idx]
            if tune_hyperparams:
                start_ktc_train = start_ktc[train_idx]
                model, best_params = _build_elasticnet_poly_tuned(
                    X_train, y_train, train_groups, start_ktc_train,
                    n_core_features=n_core, n_total_features=len(pos_features)
                )
            else:
                model = _build_elasticnet_poly(
                    n_core_features=n_core, n_total_features=len(pos_features)
                )
                model.fit(X_train, y_train)
            # Access poly features through ColumnTransformer structure
            preprocessor = model.named_steps['preprocessor']
            core_transformer = preprocessor.named_transformers_['core']
            n_poly_features = core_transformer.named_steps['poly'].n_output_features_
            n_total_features = n_poly_features + n_linear
            print(f"  Backend: ElasticNet + Mixed ({n_poly_features} poly + {n_linear} linear = {n_total_features} features)")
        else:
            # HGB ensemble (default)
            backend_name = "HGB"
            ensemble_models = []

            for eseed in ENSEMBLE_SEEDS:
                if prefer_xgb:
                    m = _try_xgb(eseed, pos)
                    if m is not None:
                        backend_name = "XGB"
                    else:
                        if eseed == ENSEMBLE_SEEDS[0]:
                            print("  XGBoost not available, falling back to HGB")
                        m = _build_hgb(eseed, pos)
                else:
                    m = _build_hgb(eseed, pos)
                ensemble_models.append(m)

            # Train all ensemble models
            for m in ensemble_models:
                m.fit(X_train, y_train)

            # Use first model for monotonic test, ensemble for predictions
            model = EnsembleModel(ensemble_models)
            print(f"  Ensemble: {len(ENSEMBLE_SEEDS)} models with seeds {ENSEMBLE_SEEDS}")

            # Monotonic smoke test (use first model in ensemble)
            _monotonic_smoke_test(ensemble_models[0], pos)

            # Train quantile models for uncertainty estimation (20th and 80th percentiles)
            for q in [0.2, 0.8]:
                q_model = _build_quantile_hgb(seed, pos, q)
                q_model.fit(X_train, y_train)
                quantile_models[q] = q_model
            print(f"  Quantile models trained (p20, p80) for uncertainty bands")

        # Raw predictions
        train_preds = model.predict(X_train)
        test_preds = model.predict(X_test)

        # Isotonic calibration using cross-validated out-of-fold predictions
        # Skip calibration for linear models (not needed)
        calibrator_dict = None
        if not no_calibration and not is_linear:
            # Generate OOF predictions so calibrator sees realistic bias
            oof_preds = np.full(len(X_train), np.nan)
            train_groups = groups[train_idx]
            n_unique = len(np.unique(train_groups))
            n_cv_folds = min(5, n_unique)

            if n_cv_folds >= 2:
                gkf = GroupKFold(n_splits=n_cv_folds)
                for cv_train, cv_val in gkf.split(X_train, y_train, train_groups):
                    cv_model = _try_xgb(seed, pos) if prefer_xgb and backend_name == "XGB" else _build_hgb(seed, pos)
                    cv_model.fit(X_train[cv_train], y_train[cv_train])
                    oof_preds[cv_val] = cv_model.predict(X_train[cv_val])

            # Fall back to train preds if OOF failed
            valid_oof = ~np.isnan(oof_preds)
            if valid_oof.sum() < GP_BUCKET_MIN_SAMPLES:
                oof_preds = train_preds
                valid_oof = np.ones(len(oof_preds), dtype=bool)

            # Global calibrator (monotone linear to preserve PPG sensitivity)
            global_cal = MonotoneLinearCalibrator(min_slope=0.01)
            global_cal.fit(oof_preds[valid_oof], y_train[valid_oof])
            calibrator_dict = {"global": global_cal}

            # Per-GP-bucket calibrators (monotone linear to preserve PPG sensitivity)
            gp_train = pos_df.iloc[train_idx]["games_played_so_far"].values
            gp_test = pos_df.iloc[test_idx]["games_played_so_far"].values

            for lo, hi in GP_BUCKETS:
                bkey = f"gp_{lo}_{hi}"
                bucket_mask = (gp_train >= lo) & (gp_train <= hi) & valid_oof
                if bucket_mask.sum() >= GP_BUCKET_MIN_SAMPLES:
                    bucket_cal = MonotoneLinearCalibrator(min_slope=0.01)
                    bucket_cal.fit(oof_preds[bucket_mask], y_train[bucket_mask])
                    calibrator_dict[bkey] = bucket_cal

            # Apply calibration to test predictions (bucket-specific, fallback to global)
            test_preds_cal = test_preds.copy()
            for i in range(len(test_preds)):
                bkey = _gp_bucket_key(gp_test[i])
                cal = calibrator_dict.get(bkey, global_cal) if bkey else global_cal
                calibrated = float(cal.predict([test_preds[i]])[0])
                if not np.isnan(calibrated):
                    test_preds_cal[i] = calibrated
            test_preds = test_preds_cal

            # Apply bucket calibration to OOF predictions (for pos_cal fitting)
            oof_preds_cal = oof_preds.copy()
            for i in range(len(oof_preds)):
                if not valid_oof[i]:
                    continue
                bkey = _gp_bucket_key(gp_train[i])
                cal = calibrator_dict.get(bkey, global_cal) if bkey else global_cal
                calibrated = float(cal.predict([oof_preds[i]])[0])
                if not np.isnan(calibrated):
                    oof_preds_cal[i] = calibrated

            # Second-stage: per-position linear calibration
            # Fit on calibrated OOF predictions (no test leakage!)
            if USE_PCT_CHANGE_TARGET:
                pre_end_ktc = start_ktc_test * (1 + test_preds)
            else:
                pre_end_ktc = start_ktc_test * np.exp(test_preds)
            pre_mae = mean_absolute_error(y_end_ktc_test, pre_end_ktc)
            pre_bias = float(np.mean(pre_end_ktc - y_end_ktc_test))

            pos_cal = MonotoneLinearCalibrator()  # Uses identity shrinkage defaults
            pos_cal.fit(oof_preds_cal[valid_oof], y_train[valid_oof])
            calibrator_dict["pos_cal"] = pos_cal

            # Apply pos_cal to test predictions
            test_preds = pos_cal.predict(test_preds)

            if USE_PCT_CHANGE_TARGET:
                post_end_ktc = start_ktc_test * (1 + test_preds)
            else:
                post_end_ktc = start_ktc_test * np.exp(test_preds)
            post_mae = mean_absolute_error(y_end_ktc_test, post_end_ktc)
            post_bias = float(np.mean(post_end_ktc - y_end_ktc_test))
            print(f"  pos_cal: MAE {pre_mae:.1f} -> {post_mae:.1f}, bias {pre_bias:+.1f} -> {post_bias:+.1f}")

        # Clip bounds: 2nd/99th percentile of y_train (asymmetric to allow upside)
        low = float(np.percentile(y_train, 2))
        high = float(np.percentile(y_train, 99))

        # Verify PPG sensitivity wasn't collapsed by calibration (HGB only)
        if calibrator_dict and not is_linear:
            _ppg_sensitivity_test(model, calibrator_dict, (low, high), pos)

        # Clip test predictions
        test_preds_clipped = np.clip(test_preds, low, high)

        # Apply KTC-aware clamp (ensures end_ktc stays within [1, 9999] domain)
        if USE_PCT_CHANGE_TARGET:
            # pct_change: end_ktc = start_ktc * (1 + pct_change)
            # For end_ktc = 9999: pct_change = (9999 - start_ktc) / start_ktc
            # For end_ktc = 1: pct_change = (1 - start_ktc) / start_ktc
            ktc_aware_upper = (9999.0 - start_ktc_test) / start_ktc_test
            ktc_aware_lower = (1.0 - start_ktc_test) / start_ktc_test
        else:
            # log_ratio: end_ktc = start_ktc * exp(log_ratio)
            ktc_aware_upper = np.log(9999.0 / start_ktc_test)
            ktc_aware_lower = np.log(1.0 / start_ktc_test)

        test_preds_clipped = np.maximum(
            ktc_aware_lower,
            np.minimum(ktc_aware_upper, test_preds_clipped),
        )

        # Convert to end_ktc for reporting
        if USE_PCT_CHANGE_TARGET:
            # pct_change: end_ktc = start_ktc * (1 + pct_change)
            test_end_ktc_preds = start_ktc_test * (1 + test_preds_clipped)
        else:
            # log_ratio: end_ktc = start_ktc * exp(log_ratio)
            test_end_ktc_preds = start_ktc_test * np.exp(test_preds_clipped)

        # Collect test-set rows (always — used by diagnostics and optional CSV export)
        test_meta = pos_df.iloc[test_idx]
        for i in range(len(test_idx)):
            actual_delta = float(y_end_ktc_test[i] - start_ktc_test[i])
            predicted_delta = float(test_end_ktc_preds[i] - start_ktc_test[i])
            age_val = test_meta.iloc[i]["age"]
            all_test_rows.append({
                "player_id": test_meta.iloc[i]["player_id"],
                "position": pos,
                "year": test_meta.iloc[i]["year"],
                "week": test_meta.iloc[i]["week"],
                "age": round(age_val, 1) if pd.notna(age_val) else None,
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
        bundle["calibrators"][pos] = calibrator_dict
        bundle["quantile_models"][pos] = quantile_models
        metrics_entry = {
            "mae": round(mae, 1),
            "r2": round(r2, 4),
            "n_train": len(X_train),
            "n_test": len(X_test),
            "backend": backend_name,
            "clip_low": round(low, 1),
            "clip_high": round(high, 1),
        }
        # Save tuned hyperparameters if available
        if best_params:
            metrics_entry["best_alpha"] = best_params.get('enet__alpha')
            metrics_entry["best_l1_ratio"] = best_params.get('enet__l1_ratio')
        bundle["metrics"][pos] = metrics_entry

    # Save per-position start_ktc medians for sentinel imputation at inference
    sentinel_impute = {}
    for pos in POSITIONS:
        pos_df = df[df["position"] == pos]
        non_sent = pos_df[pos_df["start_ktc_was_sentinel"] == 0]
        if not non_sent.empty:
            sentinel_impute[pos] = float(non_sent["start_ktc"].median())
    bundle["sentinel_impute"] = sentinel_impute

    # Save target type for inference (log_ratio vs pct_change)
    bundle["target_type"] = "pct_change" if USE_PCT_CHANGE_TARGET else "log_ratio"

    if export_csv and all_test_rows:
        csv_df = pd.DataFrame(all_test_rows)
        csv_df = csv_df.sort_values("error", key=abs, ascending=False)
        csv_df.to_csv(export_csv, index=False)
        print(f"Exported {len(csv_df)} test-set predictions to {export_csv}")

        # Sanity check: verify KTC-aware clamp is working
        max_pred = csv_df["predicted_end_ktc"].max()
        min_pred = csv_df["predicted_end_ktc"].min()
        count_over_9999 = (csv_df["predicted_end_ktc"] > 9999).sum()
        count_under_1 = (csv_df["predicted_end_ktc"] < 1).sum()
        print(f"  KTC sanity check: max={max_pred:.1f}, min={min_pred:.1f}, "
              f"count>9999={count_over_9999}, count<1={count_under_1}")
        print()

    print_diagnostics(df, all_test_rows)

    # Compute and save bias diagnostics for regression testing
    bundle["diagnostics"] = compute_bias_diagnostics(all_test_rows)

    # Train KNN adjuster for elite tier correction
    if USE_KNN_ADJUSTMENT:
        print("Training KNN elite adjuster...")
        knn_adjuster = EliteKNNAdjuster(
            k=KNN_K,
            elite_threshold=KNN_ELITE_THRESHOLD,
            blend_weight=KNN_BLEND_WEIGHT,
        )
        for pos in POSITIONS:
            pos_df = df[df["position"] == pos]
            knn_adjuster.fit(pos_df, pos)
        print(f"  {knn_adjuster}")
        bundle["knn_adjuster"] = knn_adjuster
    else:
        bundle["knn_adjuster"] = None

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
        help="Skip calibration",
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
    parser.add_argument(
        "--model-type",
        default="hgb",
        choices=MODEL_TYPES,
        help="Model type: hgb (gradient boosting), ridge (linear), elasticnet-poly (polynomial)",
    )
    parser.add_argument(
        "--tune",
        action="store_true",
        help="Run hyperparameter tuning via GridSearchCV (slower but better). Only applies to elasticnet-poly.",
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
        model_type=args.model_type,
        tune_hyperparams=args.tune,
    )


if __name__ == "__main__":
    main()
