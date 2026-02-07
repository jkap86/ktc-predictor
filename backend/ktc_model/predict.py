"""Predict end-of-season KTC value from position, games played, PPG, start KTC, and optional features."""

import numpy as np

from ktc_model.age_adjustment import apply_age_decline_adjustment, env_flag

VALID_POSITIONS = {"QB", "RB", "WR", "TE"}
GP_BUCKETS = [(1, 3), (4, 7), (8, 11), (12, 17)]

# PPG sensitivity boost by position: multiplier applied to log_ratio based on PPG
# This compensates for positions where the model has weak PPG sensitivity due to
# limited training data variance. Only applied when gp >= 8 (meaningful sample).
# Formula: log_ratio += boost * (ppg - baseline_ppg) where baseline_ppg is position average
_PPG_SENSITIVITY_BOOST = {
    "TE": {"boost": 0.035, "baseline": 8.0},  # TEs average ~8 PPG in training data
}

# Performance floor thresholds by position: (min_games, max_ppg, log_ratio_cap)
# QBs score more, so thresholds are higher; TEs score less, so thresholds are lower
_PERF_FLOOR_THRESHOLDS = {
    "QB": [(12, 2, -0.35), (12, 5, -0.25), (12, 8, -0.15),
           (8, 2, -0.20), (8, 5, -0.10)],
    "RB": [(12, 2, -0.35), (12, 4, -0.25), (12, 6, -0.15),
           (8, 2, -0.20), (8, 4, -0.10)],
    "WR": [(12, 2, -0.35), (12, 4, -0.25), (12, 6, -0.15),
           (8, 2, -0.20), (8, 4, -0.10)],
    "TE": [(12, 1, -0.35), (12, 2, -0.25), (12, 4, -0.15),
           (8, 1, -0.20), (8, 2, -0.10)],
}


def _perf_floor_log_ratio_cap(position: str, gp: float, ppg: float) -> float | None:
    """Return an upper bound for log_ratio in low-performance regimes.

    Prevents model from predicting positive growth when a player plays
    many games but scores very little (a scenario with no training data).
    """
    pos_thresholds = _PERF_FLOOR_THRESHOLDS.get(position, [])
    for min_gp, max_ppg, cap in pos_thresholds:
        if gp >= min_gp and ppg <= max_ppg:
            return cap
    return None


def _gp_bucket_key(gp: float) -> str | None:
    """Return the bucket key string for a games_played value, or None."""
    for lo, hi in GP_BUCKETS:
        if lo <= gp <= hi:
            return f"gp_{lo}_{hi}"
    return None


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

    X = np.array([[
        gp,
        ppg,
        start_ktc,
        age if age is not None else np.nan,
        weeks_missed if weeks_missed is not None else np.nan,
        draft_pick if draft_pick is not None else np.nan,
        years_remaining if years_remaining is not None else np.nan,
        was_sentinel,
    ]])

    # Predict log_ratio: log(end_ktc / start_ktc)
    pred_log_ratio = float(model.predict(X)[0])

    # Calibrate if calibrator exists
    cal_entry = calibrators.get(position)
    if cal_entry is not None:
        if isinstance(cal_entry, dict):
            # Bucketed calibrator dict: try bucket-specific, fallback to global
            bkey = _gp_bucket_key(gp)
            cal = cal_entry.get(bkey) if bkey else None
            if cal is None:
                cal = cal_entry.get("global")
        else:
            # Backward compat: bare IsotonicRegression
            cal = cal_entry
        if cal is not None:
            calibrated = float(cal.predict([pred_log_ratio])[0])
            if not np.isnan(calibrated):
                pred_log_ratio = calibrated

        # Second-stage: per-position calibration (linear)
        if isinstance(cal_entry, dict):
            pos_cal = cal_entry.get("pos_cal")
            if pos_cal is not None:
                pos_calibrated = float(pos_cal.predict([pred_log_ratio])[0])
                if not np.isnan(pos_calibrated):
                    pred_log_ratio = pos_calibrated

    # Performance floor: cap log_ratio in low-performance regimes
    # This prevents positive predictions when player has high games but very low PPG
    perf_cap = _perf_floor_log_ratio_cap(position, gp, ppg)
    if perf_cap is not None:
        pred_log_ratio = min(pred_log_ratio, perf_cap)

    # PPG sensitivity boost: compensate for positions with weak model sensitivity
    # Only apply when games played >= 8 (meaningful sample size)
    ppg_boost_config = _PPG_SENSITIVITY_BOOST.get(position)
    if ppg_boost_config is not None and gp >= 8:
        boost = ppg_boost_config["boost"]
        baseline = ppg_boost_config["baseline"]
        ppg_adjustment = boost * (ppg - baseline)
        pred_log_ratio += ppg_adjustment

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
