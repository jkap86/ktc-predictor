"""Predict end-of-season KTC value from position, games played, PPG, start KTC, and optional features."""

import numpy as np

VALID_POSITIONS = {"QB", "RB", "WR", "TE"}
GP_BUCKETS = [(1, 3), (4, 7), (8, 11), (12, 17)]


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

        # Second-stage: per-position calibration (linear or isotonic)
        if isinstance(cal_entry, dict):
            pos_cal = cal_entry.get("pos_cal")
            if pos_cal is not None:
                pos_calibrated = float(pos_cal.predict([pred_log_ratio])[0])
                if not np.isnan(pos_calibrated):
                    pred_log_ratio = pos_calibrated

    # Clip log_ratio to bounds
    bounds = clip_bounds.get(position)
    if bounds is not None:
        low, high = bounds
        pred_log_ratio = max(low, min(high, pred_log_ratio))

    # Convert to end_ktc: start_ktc * exp(log_ratio)
    end_ktc = start_ktc * np.exp(pred_log_ratio)
    delta_ktc = end_ktc - start_ktc
    return {
        "delta_ktc": round(delta_ktc, 1),
        "end_ktc": round(end_ktc, 1),
        "effective_start_ktc": round(start_ktc, 1),
    }
