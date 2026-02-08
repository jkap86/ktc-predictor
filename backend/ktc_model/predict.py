"""Predict end-of-season KTC value from position, games played, PPG, start KTC, and optional features."""

import numpy as np

from ktc_model.age_adjustment import apply_age_decline_adjustment, env_flag

VALID_POSITIONS = {"QB", "RB", "WR", "TE"}

# Prime age by position (peak dynasty value age)
PRIME_AGE = {"QB": 27, "RB": 24, "WR": 26, "TE": 27}

# Position-specific PPG baselines for z-score calculation
PPG_BASELINES = {
    "QB": {"mean": 18.0, "std": 5.0},
    "RB": {"mean": 12.0, "std": 5.0},
    "WR": {"mean": 10.0, "std": 4.0},
    "TE": {"mean": 8.0, "std": 3.5},
}


def _get_ktc_quartile(ktc: float) -> int:
    """Return KTC quartile (1-4) based on fixed boundaries."""
    if ktc < 1559:
        return 1
    elif ktc < 3085:
        return 2
    elif ktc < 4850:
        return 3
    return 4


def _age_prime_distance(age: float | None, position: str) -> float:
    """Return age distance from positional prime. Negative = before prime."""
    if age is None:
        return 0.0
    prime = PRIME_AGE.get(position, 26)
    return age - prime


def _ppg_zscore(ppg: float, position: str) -> float:
    """Return PPG as z-score relative to position average."""
    baseline = PPG_BASELINES.get(position, {"mean": 12.0, "std": 5.0})
    return (ppg - baseline["mean"]) / baseline["std"]


def _is_breakout_candidate(
    age: float | None, ktc: float, ppg: float, position: str
) -> int:
    """Binary flag: 1 if player matches breakout profile, 0 otherwise."""
    if age is None:
        return 0

    # Check KTC tier (Q2-Q3)
    ktc_q = _get_ktc_quartile(ktc)
    if ktc_q not in (2, 3):
        return 0

    # Check age (within 3 years of prime)
    prime = PRIME_AGE.get(position, 26)
    if abs(age - prime) > 3:
        return 0

    # Check PPG (above average for position)
    zscore = _ppg_zscore(ppg, position)
    if zscore < 0.5:
        return 0

    return 1


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
    sentinel_impute: dict | None = None,
) -> dict:
    """Predict end-of-season KTC value.

    Pipeline: raw model predict log_ratio -> linear calibration -> clip -> exp -> multiply start_ktc.

    Parameters
    ----------
    models : dict
        Mapping of position -> trained regressor.
    clip_bounds : dict
        Mapping of position -> (low, high) percentile bounds for log_ratio.
    calibrators : dict
        Mapping of position -> IsotonicRegression (or None).
    position : str
        One of QB, RB, WR, TE.
    gp : float
        Games played so far (>= 0).
    ppg : float
        Points per game so far (>= 0).
    start_ktc : float
        Current/start-of-season KTC value (> 0).
    age : float or None
        Player age. None -> NaN for the model.
    weeks_missed : float or None
        Weeks missed so far (injury/bye). None -> NaN for the model.
    draft_pick : float or None
        NFL draft pick number. None -> NaN for the model.
    years_remaining : float or None
        Contract years remaining. None -> NaN for the model.
    sentinel_impute : dict or None
        Mapping of position -> median start_ktc for sentinel imputation.
        If provided and start_ktc >= 9999, auto-replaces with imputed value.

    Returns
    -------
    dict
        {"delta_ktc": float, "end_ktc": float, "effective_start_ktc": float}

    Raises
    ------
    ValueError
        If position is invalid or inputs are out of range.
    KeyError
        If no model exists for the given position.
    """
    if position not in VALID_POSITIONS:
        raise ValueError(
            f"Invalid position '{position}'. Must be one of {sorted(VALID_POSITIONS)}"
        )
    if gp < 0:
        raise ValueError(f"games_played must be >= 0, got {gp}")
    if ppg < 0:
        raise ValueError(f"ppg must be >= 0, got {ppg}")
    if start_ktc <= 0:
        raise ValueError(f"start_ktc must be > 0, got {start_ktc}")
    if age is not None and age < 0:
        raise ValueError(f"age must be >= 0, got {age}")
    if weeks_missed is not None and weeks_missed < 0:
        raise ValueError(f"weeks_missed must be >= 0, got {weeks_missed}")
    if draft_pick is not None and draft_pick < 1:
        raise ValueError(f"draft_pick must be >= 1, got {draft_pick}")
    if years_remaining is not None and years_remaining < 0:
        raise ValueError(f"years_remaining must be >= 0, got {years_remaining}")

    if position not in models:
        raise KeyError(f"No model available for position '{position}'")

    model = models[position]

    # Auto-detect and replace sentinel start_ktc values
    was_sentinel = 0
    if start_ktc >= 9999 and sentinel_impute and position in sentinel_impute:
        was_sentinel = 1
        start_ktc = sentinel_impute[position]

    # Compute engineered features
    ktc_quartile = _get_ktc_quartile(start_ktc)
    age_prime_dist = _age_prime_distance(age, position)
    ppg_z = _ppg_zscore(ppg, position)
    breakout_flag = _is_breakout_candidate(age, start_ktc, ppg, position)

    X = np.array([[
        gp,
        ppg,
        start_ktc,
        age if age is not None else np.nan,
        weeks_missed if weeks_missed is not None else np.nan,
        draft_pick if draft_pick is not None else np.nan,
        years_remaining if years_remaining is not None else np.nan,
        was_sentinel,
        ktc_quartile,
        age_prime_dist,
        ppg_z,
        breakout_flag,
    ]])

    # Predict log_ratio: log(end_ktc / start_ktc)
    # Note: ElasticNet+Poly model handles feature interactions internally,
    # so no post-hoc calibration or adjustments are needed
    pred_log_ratio = float(model.predict(X)[0])

    # Clip log_ratio to percentile bounds from training data
    # This prevents extreme predictions outside the observed distribution
    bounds = clip_bounds.get(position)
    if bounds is not None:
        low, high = bounds
        pred_log_ratio = max(low, min(high, pred_log_ratio))

    # Age decline adjustment (feature-flagged)
    if env_flag("KTC_ENABLE_AGE_DECLINE_ADJ", default=False):
        pred_log_ratio = apply_age_decline_adjustment(pred_log_ratio, age, position)

    # KTC-aware log_ratio bounds: prevent flatline at domain boundaries
    # This ensures end_ktc can never mathematically exceed [1, 9999]
    ktc_aware_upper = np.log(9999.0 / start_ktc)
    ktc_aware_lower = np.log(1.0 / start_ktc)
    pred_log_ratio = max(ktc_aware_lower, min(ktc_aware_upper, pred_log_ratio))

    # KTC domain bounds
    KTC_MIN = 1.0
    KTC_MAX = 9999.0

    # Convert to end_ktc: start_ktc * exp(log_ratio)
    raw_end_ktc = start_ktc * np.exp(pred_log_ratio)

    # Clamp to valid KTC domain
    end_ktc = max(KTC_MIN, min(KTC_MAX, raw_end_ktc))

    # Recompute delta using clamped value
    delta_ktc = end_ktc - start_ktc

    # Track if clamping occurred (for optional UI indication)
    capped_high = raw_end_ktc > KTC_MAX
    capped_low = raw_end_ktc < KTC_MIN

    return {
        "delta_ktc": round(delta_ktc, 1),
        "end_ktc": round(end_ktc, 1),
        "effective_start_ktc": round(start_ktc, 1),
        "capped_high": capped_high,
        "capped_low": capped_low,
    }
