"""Predict end-of-season KTC value from position, games played, and PPG."""

import numpy as np

VALID_POSITIONS = {"QB", "RB", "WR", "TE"}


def predict_end_ktc(
    models: dict,
    clip_bounds: dict,
    calibrators: dict,
    position: str,
    gp: float,
    ppg: float,
) -> float:
    """Predict end-of-season KTC value.

    Pipeline: raw model predict -> isotonic calibration -> clip to bounds.

    Parameters
    ----------
    models : dict
        Mapping of position -> trained regressor.
    clip_bounds : dict
        Mapping of position -> (low, high) percentile bounds.
    calibrators : dict
        Mapping of position -> IsotonicRegression (or None).
    position : str
        One of QB, RB, WR, TE.
    gp : float
        Games played so far (>= 0).
    ppg : float
        Points per game so far (>= 0).

    Returns
    -------
    float
        Predicted end-of-season KTC value.

    Raises
    ------
    ValueError
        If position is invalid or gp/ppg are negative.
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

    if position not in models:
        raise KeyError(f"No model available for position '{position}'")

    model = models[position]
    X = np.array([[gp, ppg]])

    raw_pred = float(model.predict(X)[0])

    # Calibrate if calibrator exists
    calibrator = calibrators.get(position)
    if calibrator is not None:
        calibrated = float(calibrator.predict([raw_pred])[0])
        if not np.isnan(calibrated):
            raw_pred = calibrated

    # Clip to bounds
    bounds = clip_bounds.get(position)
    if bounds is not None:
        low, high = bounds
        raw_pred = max(low, min(high, raw_pred))

    return round(raw_pred, 1)
