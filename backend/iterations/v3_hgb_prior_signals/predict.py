"""v3 inference: v1's prediction pipeline extended to 26 features.

Duplicates v1's predict_end_ktc (rather than shimming) because v1 builds its
feature vector inline with a fixed 20-slot layout. The postprocessing steps
(KNN adjustment, residual correction, extreme shrinkage, clipping, KTC-aware
bounds, domain clamp) are identical — we import v1's helpers directly.
"""

import numpy as np

from iterations.v1_hgb_baseline.age_adjustment import (
    apply_age_decline_adjustment,
    env_flag,
)
from iterations.v1_hgb_baseline.predict import (
    PPG_BASELINES,
    PRIME_AGE,
    VALID_POSITIONS,
    _age_prime_distance,
    _get_ktc_quartile,
    _is_breakout_candidate,
    apply_extreme_shrinkage,
    apply_residual_correction,
)

# v3 feature list: v1 QB superset (20) + 6 prior-behavioral = 26
_V1_CORE_FEATURES = [
    "games_played_so_far",
    "ppg_so_far",
    "weeks_missed_so_far",
    "draft_pick",
    "years_remaining",
    "start_ktc_quartile",
    "age_prime_distance",
    "is_breakout_candidate",
]

_V1_LINEAR_FEATURES = [
    "start_ktc",
    "start_ktc_was_sentinel",
    "ktc_yoy_log",
    "ktc_peak_drawdown",
    "has_prior_season",
    "prior_ppg",
    "ppg_yoy_log",
    "has_prior_ppg",
    "apy_cap_pct",
    "is_contract_year",
    "apy_position_rank",
    "has_contract_data",
]

_PRIOR_BEHAVIORAL_FEATURES = [
    "prior_weekly_fp_cv",
    "prior_boom_rate",
    "prior_bust_rate",
    "prior_snap_pct",
    "prior_ktc_volatility",
    "has_prior_behavioral",
]


def get_expected_features(position: str) -> list[str]:
    """Get the expected feature list for v3 (same for all positions)."""
    return _V1_CORE_FEATURES + _V1_LINEAR_FEATURES + _PRIOR_BEHAVIORAL_FEATURES


EXPECTED_FEATURES = get_expected_features("QB")


def validate_feature_contract(
    saved_features: list[str] | dict[str, list[str]] | None,
    position: str | None = None,
) -> None:
    """Verify the model bundle's feature_names match v3's expectations."""
    if saved_features is None:
        return

    if isinstance(saved_features, dict):
        for pos, features in saved_features.items():
            if position is not None and pos != position:
                continue
            expected = get_expected_features(pos)
            if features != expected:
                raise ValueError(
                    f"v3 feature contract mismatch for {pos}!\n"
                    f"Model expects: {features}\n"
                    f"v3 predict.py has: {expected}"
                )
        return

    if saved_features != EXPECTED_FEATURES:
        raise ValueError(
            f"v3 feature contract mismatch!\n"
            f"Model expects: {saved_features}\n"
            f"v3 predict.py has: {EXPECTED_FEATURES}"
        )


def predict_end_ktc(
    models: dict,
    clip_bounds: dict,
    calibrators: dict,
    position: str,
    gp: float,
    ppg: float,
    start_ktc: float,
    age: float | None = None,
    weeks_missed: float | None = None,
    draft_pick: float | None = None,
    years_remaining: float | None = None,
    prior_end_ktc: float | None = None,
    max_ktc_prior: float | None = None,
    prior_ppg: float | None = None,
    apy_cap_pct: float | None = None,
    is_contract_year: float | None = None,
    apy_position_rank: float | None = None,
    # v3 additions (all optional; NaN-impute falls back to model's imputer)
    prior_weekly_fp_cv: float | None = None,
    prior_boom_rate: float | None = None,
    prior_bust_rate: float | None = None,
    prior_snap_pct: float | None = None,
    prior_ktc_volatility: float | None = None,
    sentinel_impute: dict | None = None,
    residual_correction: dict | None = None,
    target_type: str = "log_ratio",
    knn_adjuster=None,
    **_unused_kwargs,
) -> dict:
    """Predict end-of-season KTC value using v3's 26-feature model."""
    if position not in VALID_POSITIONS:
        raise ValueError(
            f"Invalid position '{position}'. Must be one of {sorted(VALID_POSITIONS)}"
        )
    if gp < 0:
        raise ValueError(f"games_played must be >= 0, got {gp}")
    if ppg < 0:
        ppg = 0.0
    if start_ktc <= 0:
        raise ValueError(f"start_ktc must be > 0, got {start_ktc}")
    if age is not None and age < 0:
        raise ValueError(f"age must be >= 0, got {age}")

    if position not in models:
        raise KeyError(f"No model available for position '{position}'")

    model = models[position]

    # Sentinel imputation
    was_sentinel = 0
    if start_ktc >= 9999 and sentinel_impute and position in sentinel_impute:
        was_sentinel = 1
        start_ktc = sentinel_impute[position]

    # Engineered features (same helpers as v1)
    ktc_quartile = _get_ktc_quartile(start_ktc)
    age_prime_dist = _age_prime_distance(age, position)
    breakout_flag = _is_breakout_candidate(age, start_ktc, ppg, position)

    # Core features (8) — unchanged from v1
    core_features = [
        gp,
        ppg,
        weeks_missed if weeks_missed is not None else np.nan,
        draft_pick if draft_pick is not None else np.nan,
        years_remaining if years_remaining is not None else np.nan,
        ktc_quartile,
        age_prime_dist,
        breakout_flag,
    ]

    # Base linear (2)
    linear_features = [start_ktc, was_sentinel]

    # Prior-season KTC (3)
    if prior_end_ktc is not None and prior_end_ktc > 0:
        ktc_yoy_log = float(np.clip(np.log(start_ktc / prior_end_ktc), -0.7, 0.7))
        has_prior_season = 1
    else:
        ktc_yoy_log = np.nan
        has_prior_season = 0
    if max_ktc_prior is not None and max_ktc_prior > 0:
        ktc_peak_drawdown = float(np.log(start_ktc / max_ktc_prior))
    else:
        ktc_peak_drawdown = np.nan
    linear_features.extend([ktc_yoy_log, ktc_peak_drawdown, has_prior_season])

    # Prior-season PPG (3)
    if prior_ppg is not None and prior_ppg > 0 and ppg > 0:
        eps = 0.1
        ppg_yoy_log = float(np.clip(np.log((ppg + eps) / (prior_ppg + eps)), -1.0, 1.0))
        has_prior_ppg = 1
        prior_ppg_val = prior_ppg
    else:
        prior_ppg_val = np.nan
        ppg_yoy_log = np.nan
        has_prior_ppg = 0
    linear_features.extend([prior_ppg_val, ppg_yoy_log, has_prior_ppg])

    # Contract (4)
    has_contract_data = 1 if apy_cap_pct is not None else 0
    linear_features.extend([
        apy_cap_pct if apy_cap_pct is not None else np.nan,
        is_contract_year if is_contract_year is not None else np.nan,
        apy_position_rank if apy_position_rank is not None else np.nan,
        has_contract_data,
    ])

    # v3 prior-behavioral (6)
    has_prior_behavioral = 1 if prior_weekly_fp_cv is not None else 0
    linear_features.extend([
        prior_weekly_fp_cv if prior_weekly_fp_cv is not None else np.nan,
        prior_boom_rate if prior_boom_rate is not None else np.nan,
        prior_bust_rate if prior_bust_rate is not None else np.nan,
        prior_snap_pct if prior_snap_pct is not None else np.nan,
        prior_ktc_volatility if prior_ktc_volatility is not None else np.nan,
        has_prior_behavioral,
    ])

    X = np.array([core_features + linear_features])

    pred_log_ratio = float(model.predict(X)[0])

    # KNN elite adjustment
    if knn_adjuster is not None:
        pred_log_ratio = knn_adjuster.adjust(
            position=position,
            model_log_ratio=pred_log_ratio,
            age_prime_dist=age_prime_dist,
            ppg=ppg,
            start_ktc=start_ktc,
            gp=gp,
        )

    # Residual correction
    if env_flag("KTC_ENABLE_RESIDUAL_CORRECTION", default=True):
        pred_log_ratio = apply_residual_correction(
            pred_log_ratio, start_ktc, position, residual_correction
        )

    # Extreme shrinkage
    if env_flag("KTC_ENABLE_EXTREME_SHRINKAGE", default=True):
        pred_log_ratio = apply_extreme_shrinkage(pred_log_ratio)

    # Clip to percentile bounds
    bounds = clip_bounds.get(position)
    if bounds is not None:
        low, high = bounds
        pred_log_ratio = max(low, min(high, pred_log_ratio))

    # Age decline adjustment
    if env_flag("KTC_ENABLE_AGE_DECLINE_ADJ", default=False):
        pred_log_ratio = apply_age_decline_adjustment(pred_log_ratio, age, position)

    # KTC-aware bounds
    if target_type == "pct_change":
        ktc_aware_upper = (9999.0 - start_ktc) / start_ktc
        ktc_aware_lower = (1.0 - start_ktc) / start_ktc
    else:
        ktc_aware_upper = np.log(9999.0 / start_ktc)
        ktc_aware_lower = np.log(1.0 / start_ktc)
    pred_log_ratio = max(ktc_aware_lower, min(ktc_aware_upper, pred_log_ratio))

    KTC_MIN = 1.0
    KTC_MAX = 9999.0

    if target_type == "pct_change":
        raw_end_ktc = start_ktc * (1 + pred_log_ratio)
    else:
        raw_end_ktc = start_ktc * np.exp(pred_log_ratio)

    end_ktc = max(KTC_MIN, min(KTC_MAX, raw_end_ktc))
    delta_ktc = end_ktc - start_ktc

    return {
        "delta_ktc": round(delta_ktc, 1),
        "end_ktc": round(end_ktc, 1),
        "effective_start_ktc": round(start_ktc, 1),
        "capped_high": raw_end_ktc > KTC_MAX,
        "capped_low": raw_end_ktc < KTC_MIN,
    }
