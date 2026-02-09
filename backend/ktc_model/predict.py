"""Predict end-of-season KTC value from position, games played, PPG, start KTC, and optional features."""

import numpy as np

from ktc_model.age_adjustment import apply_age_decline_adjustment, env_flag

VALID_POSITIONS = {"QB", "RB", "WR", "TE"}

# ============================================================================
# POSITION-SPECIFIC FEATURE CONTRACTS
# ============================================================================
# QB, WR, TE use prior-season KTC features (trajectory signal is stable).
# RB does NOT use these features (variance dominates at elite tier, features hurt more than help).

# Core features (polynomial expansion) - same for all positions
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

# Base linear features (all positions)
_BASE_LINEAR_FEATURES = [
    "start_ktc",
    "start_ktc_was_sentinel",
]

# Prior-season KTC features (QB, WR, TE - stable trajectory positions)
_PRIOR_SEASON_FEATURES = [
    "ktc_yoy_log",          # log(start_ktc / prior_end_ktc), clipped to [-0.7, 0.7]
    "ktc_peak_drawdown",    # log(start_ktc / max_ktc_prior)
    "has_prior_season",     # 1 if prior season data exists
]

# Prior-season PPG features (QB, WR, TE - captures performance trajectory)
_PRIOR_PPG_FEATURES = [
    "prior_ppg",            # Prior season's PPG (absolute baseline)
    "ppg_yoy_log",          # log(ppg_so_far / prior_ppg), clipped to [-1.0, 1.0]
    "has_prior_ppg",        # 1 if valid prior PPG exists
]

# Contract features (all positions when USE_CONTRACT_FEATURES=True in train.py)
_CONTRACT_FEATURES = [
    "apy_cap_pct",          # APY as % of salary cap (0.0 to ~0.25)
    "is_contract_year",     # 1 if in final year, 0 otherwise
    "apy_position_rank",    # Percentile of APY within position (0-1)
    "has_contract_data",    # 1 if contract info exists, else 0
]


def get_expected_features(position: str) -> list[str]:
    """Get the expected feature list for a specific position."""
    linear_features = _BASE_LINEAR_FEATURES.copy()
    # Prior-season features for stable trajectory positions (not RB)
    if position in ("QB", "WR", "TE"):
        linear_features.extend(_PRIOR_SEASON_FEATURES)
        linear_features.extend(_PRIOR_PPG_FEATURES)
    # Contract features are used for all positions
    linear_features.extend(_CONTRACT_FEATURES)
    return _CORE_FEATURES + linear_features


# Legacy: EXPECTED_FEATURES for backwards compatibility (QB superset)
EXPECTED_FEATURES = get_expected_features("QB")


def validate_feature_contract(
    saved_features: list[str] | dict[str, list[str]] | None,
    position: str | None = None,
) -> None:
    """Validate that saved model feature names match expected features.

    Parameters
    ----------
    saved_features : list[str] or dict[str, list[str]] or None
        Either a single feature list (legacy) or a dict mapping position -> features.
    position : str or None
        Position to validate. If None and saved_features is a dict, validates all.

    Raises
    ------
    ValueError
        If feature contract mismatch is detected.
    """
    if saved_features is None:
        # No feature contract saved (legacy model) - skip validation
        return

    # Handle dict of position -> features (new format)
    if isinstance(saved_features, dict):
        if position is not None:
            # Validate specific position
            if position in saved_features:
                expected = get_expected_features(position)
                if saved_features[position] != expected:
                    raise ValueError(
                        f"Feature contract mismatch for {position}!\n"
                        f"Model expects: {saved_features[position]}\n"
                        f"Predict.py has: {expected}"
                    )
        else:
            # Validate all positions
            for pos, features in saved_features.items():
                expected = get_expected_features(pos)
                if features != expected:
                    raise ValueError(
                        f"Feature contract mismatch for {pos}!\n"
                        f"Model expects: {features}\n"
                        f"Predict.py has: {expected}"
                    )
        return

    # Handle legacy single list format (validate against QB superset)
    if saved_features != EXPECTED_FEATURES:
        raise ValueError(
            f"Feature contract mismatch!\n"
            f"Model expects: {saved_features}\n"
            f"Predict.py has: {EXPECTED_FEATURES}\n"
            f"Update EXPECTED_FEATURES in predict.py to match the trained model."
        )

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


def apply_residual_correction(
    log_ratio: float,
    start_ktc: float,
    position: str,
    residual_correction: dict | None,
) -> float:
    """Apply position-specific residual correction in log-ratio space.

    Supports multiple correction types (can be combined as an array):
    - "hinge": Piecewise linear with a hinge point
      correction = b0 + b1 * z + b2 * max(z - hinge_z, 0)
    - "linear": Simple linear correction
      correction = b0 + b1 * z
    - "band": Fixed offset within a KTC range
      correction = offset if low_ktc <= start_ktc < high_ktc else 0
    - "riser_band": Fixed offset within a KTC range, ONLY if model predicts rise
      correction = offset if low_ktc <= start_ktc < high_ktc AND log_ratio > 0 else 0

    Where z = (start_ktc - ktc_mean) / ktc_std

    Parameters
    ----------
    log_ratio : float
        Raw predicted log_ratio from the model.
    start_ktc : float
        The starting KTC value (for computing z).
    position : str
        Player position (QB, RB, WR, TE).
    residual_correction : dict or None
        Mapping of position -> correction parameters (dict or list of dicts).

    Returns
    -------
    float
        Corrected log_ratio.
    """
    if residual_correction is None:
        return log_ratio

    if position not in residual_correction:
        return log_ratio

    entry = residual_correction[position]

    # Support both single dict and list of dicts
    corrections = entry if isinstance(entry, list) else [entry]

    total_correction = 0.0
    for params in corrections:
        correction_type = params.get("type", "linear")

        if correction_type == "hinge":
            ktc_mean = params.get("ktc_mean", 0.0)
            ktc_std = params.get("ktc_std", 1.0)
            b0 = params.get("b0", 0.0)
            b1 = params.get("b1", 0.0)
            b2 = params.get("b2", 0.0)
            hinge_z = params.get("hinge_z", 0.0)
            z = (start_ktc - ktc_mean) / ktc_std
            total_correction += b0 + b1 * z + b2 * max(z - hinge_z, 0.0)

        elif correction_type == "band":
            low_ktc = params.get("low_ktc", 0)
            high_ktc = params.get("high_ktc", float("inf"))
            offset = params.get("offset", 0.0)
            if low_ktc <= start_ktc < high_ktc:
                total_correction += offset

        elif correction_type == "riser_band":
            # Elite tier riser correction: only apply if model predicts rise
            low_ktc = params.get("low_ktc", 0)
            high_ktc = params.get("high_ktc", float("inf"))
            offset = params.get("offset", 0.0)
            # Only apply if in KTC range AND model predicts positive change
            if low_ktc <= start_ktc < high_ktc and log_ratio > 0:
                total_correction += offset

        elif correction_type == "linear":
            ktc_mean = params.get("ktc_mean", 0.0)
            ktc_std = params.get("ktc_std", 1.0)
            b0 = params.get("b0", 0.0)
            b1 = params.get("b1", 0.0)
            z = (start_ktc - ktc_mean) / ktc_std
            total_correction += b0 + b1 * z

    return log_ratio + total_correction


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
    sentinel_impute: dict | None = None,
    residual_correction: dict | None = None,
    target_type: str = "log_ratio",
    knn_adjuster=None,
) -> dict:
    """Predict end-of-season KTC value.

    Pipeline: raw model predict log_ratio -> residual correction -> clip -> exp -> multiply start_ktc.

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
    prior_end_ktc : float or None
        End KTC from prior season. Used to compute YoY trajectory.
        None for first-year players or unknown history.
    max_ktc_prior : float or None
        Maximum KTC achieved in all prior seasons. Used to compute peak drawdown.
        None for first-year players or unknown history.
    prior_ppg : float or None
        PPG from prior season. Used to compute YoY performance trajectory (QB only).
        None for first-year players or unknown history.
    sentinel_impute : dict or None
        Mapping of position -> median start_ktc for sentinel imputation.
        If provided and start_ktc >= 9999, auto-replaces with imputed value.
    residual_correction : dict or None
        Mapping of position -> residual correction parameters.
        Applied when KTC_ENABLE_RESIDUAL_CORRECTION env flag is set.
    knn_adjuster : EliteKNNAdjuster or None
        KNN-based adjustment for elite tier predictions. If provided,
        blends model prediction with outcomes of similar historical players.

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
    breakout_flag = _is_breakout_candidate(age, start_ktc, ppg, position)

    # Build feature vector
    # Order: core features (polynomial expansion), then linear features (passthrough)
    # QB uses additional prior-season KTC features; RB/WR/TE do not

    # Core features (same for all positions)
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

    # Base linear features (same for all positions)
    linear_features = [
        start_ktc,
        was_sentinel,
    ]

    # Prior-season KTC features for stable trajectory positions (QB, WR, TE)
    # RB excluded due to high variance at elite tier (injuries, role shocks)
    if position in ("QB", "WR", "TE"):
        # Compute prior-season KTC features
        if prior_end_ktc is not None and prior_end_ktc > 0:
            ktc_yoy_log = float(np.clip(np.log(start_ktc / prior_end_ktc), -0.7, 0.7))
            has_prior_season = 1
        else:
            ktc_yoy_log = np.nan  # Will be median-imputed by model
            has_prior_season = 0

        if max_ktc_prior is not None and max_ktc_prior > 0:
            ktc_peak_drawdown = float(np.log(start_ktc / max_ktc_prior))
        else:
            ktc_peak_drawdown = np.nan  # Will be median-imputed by model

        linear_features.extend([ktc_yoy_log, ktc_peak_drawdown, has_prior_season])

        # Prior-season PPG features (performance trajectory)
        if prior_ppg is not None and prior_ppg > 0 and ppg > 0:
            eps = 0.1
            ppg_yoy_log = float(np.clip(np.log((ppg + eps) / (prior_ppg + eps)), -1.0, 1.0))
            has_prior_ppg = 1
        else:
            prior_ppg_val = np.nan  # Will be median-imputed by model
            ppg_yoy_log = np.nan
            has_prior_ppg = 0
            prior_ppg = prior_ppg_val  # For feature vector

        linear_features.extend([prior_ppg if prior_ppg is not None else np.nan, ppg_yoy_log, has_prior_ppg])

    # Contract features (all positions when model was trained with USE_CONTRACT_FEATURES=True)
    # These help distinguish rookie deals vs established vets, contract year performers, etc.
    has_contract_data = 1 if apy_cap_pct is not None else 0
    linear_features.extend([
        apy_cap_pct if apy_cap_pct is not None else np.nan,
        is_contract_year if is_contract_year is not None else np.nan,
        apy_position_rank if apy_position_rank is not None else np.nan,
        has_contract_data,
    ])

    X = np.array([core_features + linear_features])

    # Predict log_ratio: log(end_ktc / start_ktc)
    # Note: ElasticNet+Poly model handles feature interactions internally,
    # so no post-hoc calibration or adjustments are needed
    pred_log_ratio = float(model.predict(X)[0])

    # KNN adjustment for elite tier (reduces under-prediction of risers)
    # Applied before residual correction and clipping
    if knn_adjuster is not None:
        pred_log_ratio = knn_adjuster.adjust(
            position=position,
            model_log_ratio=pred_log_ratio,
            age_prime_dist=age_prime_dist,
            ppg=ppg,
            start_ktc=start_ktc,
            gp=gp,
        )

    # Residual correction (enabled by default for RB hinge correction)
    # Applies position-specific bias correction learned from training residuals
    if env_flag("KTC_ENABLE_RESIDUAL_CORRECTION", default=True):
        pred_log_ratio = apply_residual_correction(
            pred_log_ratio, start_ktc, position, residual_correction
        )

    # Clip log_ratio to percentile bounds from training data
    # This prevents extreme predictions outside the observed distribution
    bounds = clip_bounds.get(position)
    if bounds is not None:
        low, high = bounds
        pred_log_ratio = max(low, min(high, pred_log_ratio))

    # Age decline adjustment (feature-flagged)
    if env_flag("KTC_ENABLE_AGE_DECLINE_ADJ", default=False):
        pred_log_ratio = apply_age_decline_adjustment(pred_log_ratio, age, position)

    # KTC-aware bounds: prevent flatline at domain boundaries
    # This ensures end_ktc can never mathematically exceed [1, 9999]
    if target_type == "pct_change":
        # pct_change: end_ktc = start_ktc * (1 + pct_change)
        ktc_aware_upper = (9999.0 - start_ktc) / start_ktc
        ktc_aware_lower = (1.0 - start_ktc) / start_ktc
    else:
        # log_ratio: end_ktc = start_ktc * exp(log_ratio)
        ktc_aware_upper = np.log(9999.0 / start_ktc)
        ktc_aware_lower = np.log(1.0 / start_ktc)
    pred_log_ratio = max(ktc_aware_lower, min(ktc_aware_upper, pred_log_ratio))

    # KTC domain bounds
    KTC_MIN = 1.0
    KTC_MAX = 9999.0

    # Convert to end_ktc based on target type
    if target_type == "pct_change":
        raw_end_ktc = start_ktc * (1 + pred_log_ratio)
    else:
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
