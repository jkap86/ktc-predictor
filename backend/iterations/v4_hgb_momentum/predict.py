"""v4 inference: v3 + KTC momentum + position-specific efficiency features.

Feature layout per position:
  QB:    v1(20) + momentum(4) + [prior_passing_tds, prior_interceptions, has] = 27
  RB:    v3(26) + momentum(4) + [prior_carries, prior_red_zone_touches, has] = 33
  WR/TE: v3(26) + momentum(4) + [prior_targets, prior_red_zone_targets, has] = 33
  TE:    v1(20) + momentum(4) + [prior_targets, prior_red_zone_targets, has] = 27
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

# Positions that receive the 6 prior-behavioral features.
# Must stay in sync with train._POSITIONS_WITH_PRIOR_BEHAVIORAL.
_POSITIONS_WITH_PRIOR_BEHAVIORAL = {"RB", "WR"}

# v4 additions
_MOMENTUM_FEATURES = [
    "ktc_30d_trend",
    "ktc_90d_trend",
    "momentum_ratio",
    "max_games_missed_streak",
]

_POSITION_STAT_FEATURES = {
    "QB": ["prior_passing_tds", "prior_interceptions", "prior_completion_rate",
           "prior_rushing_yards", "prior_pass_sacks"],
    "RB": ["prior_carries", "prior_red_zone_touches", "prior_yards_per_carry",
           "prior_receiving_yards", "prior_rushing_tds"],
    "WR": ["prior_targets", "prior_red_zone_targets", "prior_yards_per_target",
           "prior_air_yards_per_target", "prior_receiving_tds", "prior_drop_rate"],
    "TE": ["prior_targets", "prior_red_zone_targets", "prior_yards_per_target",
           "prior_receiving_tds", "prior_drop_rate"],
}


_POSITIONS_WITH_V4_FEATURES = {"QB", "WR"}

_TEAM_FEATURES = ["qb_ktc", "team_total_ktc", "positional_competition"]
_POSITIONS_WITH_TEAM_FEATURES = {"WR"}


def get_expected_features(position: str) -> list[str]:
    """Get the expected feature list for v4 for a given position."""
    base = _V1_CORE_FEATURES + _V1_LINEAR_FEATURES
    if position in _POSITIONS_WITH_PRIOR_BEHAVIORAL:
        base = base + _PRIOR_BEHAVIORAL_FEATURES
    extras: list[str] = []
    if position in _POSITIONS_WITH_V4_FEATURES:
        extras.extend(_MOMENTUM_FEATURES)
    pos_stats = _POSITION_STAT_FEATURES.get(position, [])
    if pos_stats:
        extras.extend(pos_stats)
        extras.append("has_prior_position_stats")
    if position in _POSITIONS_WITH_TEAM_FEATURES:
        extras.extend(_TEAM_FEATURES)
    if not extras:
        return base
    return base + extras


EXPECTED_FEATURES = get_expected_features("RB")  # largest layout


def validate_feature_contract(
    saved_features: list[str] | dict[str, list[str]] | None,
    position: str | None = None,
) -> None:
    """Verify the model bundle's feature_names match v4's expectations."""
    if saved_features is None:
        return

    if isinstance(saved_features, dict):
        for pos, features in saved_features.items():
            if position is not None and pos != position:
                continue
            expected = get_expected_features(pos)
            if features != expected:
                raise ValueError(
                    f"v4 feature contract mismatch for {pos}!\n"
                    f"Model expects: {features}\n"
                    f"v3 predict.py has: {expected}"
                )
        return

    if saved_features != EXPECTED_FEATURES:
        raise ValueError(
            f"v4 feature contract mismatch!\n"
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
    # v3 additions
    prior_weekly_fp_cv: float | None = None,
    prior_boom_rate: float | None = None,
    prior_bust_rate: float | None = None,
    prior_snap_pct: float | None = None,
    prior_ktc_volatility: float | None = None,
    # v4 additions: momentum (current season)
    ktc_30d_trend: float | None = None,
    ktc_90d_trend: float | None = None,
    momentum_ratio: float | None = None,
    max_games_missed_streak: float | None = None,
    # v4 additions: position-specific prior stats
    prior_passing_tds: float | None = None,
    prior_interceptions: float | None = None,
    prior_carries: float | None = None,
    prior_red_zone_touches: float | None = None,
    prior_targets: float | None = None,
    prior_red_zone_targets: float | None = None,
    # v4 efficiency features
    prior_completion_rate: float | None = None,
    prior_rushing_yards: float | None = None,
    prior_pass_sacks: float | None = None,
    prior_yards_per_carry: float | None = None,
    prior_receiving_yards: float | None = None,
    prior_rushing_tds: float | None = None,
    prior_yards_per_target: float | None = None,
    prior_air_yards_per_target: float | None = None,
    prior_receiving_tds: float | None = None,
    prior_drop_rate: float | None = None,
    # v4 team context (WR only)
    qb_ktc: float | None = None,
    team_total_ktc: float | None = None,
    positional_competition: float | None = None,
    sentinel_impute: dict | None = None,
    residual_correction: dict | None = None,
    target_type: str = "log_ratio",
    knn_adjuster=None,
    quantile_models: dict | None = None,
    **_unused_kwargs,
) -> dict:
    """Predict end-of-season KTC value using v4's feature model."""
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

    # v3 prior-behavioral (6) — RB/WR only
    if position in _POSITIONS_WITH_PRIOR_BEHAVIORAL:
        has_prior_behavioral = 1 if prior_weekly_fp_cv is not None else 0
        linear_features.extend([
            prior_weekly_fp_cv if prior_weekly_fp_cv is not None else np.nan,
            prior_boom_rate if prior_boom_rate is not None else np.nan,
            prior_bust_rate if prior_bust_rate is not None else np.nan,
            prior_snap_pct if prior_snap_pct is not None else np.nan,
            prior_ktc_volatility if prior_ktc_volatility is not None else np.nan,
            has_prior_behavioral,
        ])

    # v4 momentum (QB/WR only)
    if position in _POSITIONS_WITH_V4_FEATURES:
        linear_features.extend([
            ktc_30d_trend if ktc_30d_trend is not None else 0.0,
            ktc_90d_trend if ktc_90d_trend is not None else 0.0,
            momentum_ratio if momentum_ratio is not None else 0.0,
            max_games_missed_streak if max_games_missed_streak is not None else 0.0,
        ])

    # Position-specific prior efficiency stats (all positions)
    _pos_features = _POSITION_STAT_FEATURES.get(position, [])
    if _pos_features:
        _pos_kwargs = {
            "prior_passing_tds": prior_passing_tds,
            "prior_interceptions": prior_interceptions,
            "prior_completion_rate": prior_completion_rate,
            "prior_rushing_yards": prior_rushing_yards,
            "prior_pass_sacks": prior_pass_sacks,
            "prior_carries": prior_carries,
            "prior_red_zone_touches": prior_red_zone_touches,
            "prior_yards_per_carry": prior_yards_per_carry,
            "prior_receiving_yards": prior_receiving_yards,
            "prior_rushing_tds": prior_rushing_tds,
            "prior_targets": prior_targets,
            "prior_red_zone_targets": prior_red_zone_targets,
            "prior_yards_per_target": prior_yards_per_target,
            "prior_air_yards_per_target": prior_air_yards_per_target,
            "prior_receiving_tds": prior_receiving_tds,
            "prior_drop_rate": prior_drop_rate,
        }
        has_ps = 0
        for fname in _pos_features:
            val = _pos_kwargs.get(fname)
            linear_features.append(val if val is not None else np.nan)
            if val is not None:
                has_ps = 1
        linear_features.append(has_ps)

    # v4 team context (WR only)
    if position in _POSITIONS_WITH_TEAM_FEATURES:
        linear_features.extend([
            qb_ktc if qb_ktc is not None else np.nan,
            team_total_ktc if team_total_ktc is not None else np.nan,
            positional_competition if positional_competition is not None else np.nan,
        ])

    X = np.array([core_features + linear_features])

    raw_pred_log_ratio = float(model.predict(X)[0])
    pred_log_ratio = raw_pred_log_ratio

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

    result = {
        "delta_ktc": round(delta_ktc, 1),
        "end_ktc": round(end_ktc, 1),
        "effective_start_ktc": round(start_ktc, 1),
        "capped_high": raw_end_ktc > KTC_MAX,
        "capped_low": raw_end_ktc < KTC_MIN,
    }

    # Confidence bands from the p20/p80 quantile models, if available.
    # The quantile models predict raw log_ratios without the central model's
    # KNN / residual correction / shrinkage adjustments. To produce a band
    # that actually brackets the reported central prediction, we preserve the
    # band WIDTH from the quantile models and SHIFT by the same delta the
    # central got from its adjustments.
    if quantile_models and position in quantile_models:
        q_dict = quantile_models[position]
        adjustment_delta = pred_log_ratio - raw_pred_log_ratio

        def _bound_end_ktc(q_log: float) -> float:
            # Apply the central's adjustment shift, then the same clipping / KTC-aware
            # bounds / domain clamp the central went through.
            q_log = q_log + adjustment_delta
            q_log = max(ktc_aware_lower, min(ktc_aware_upper, q_log))
            if bounds is not None:
                q_log = max(bounds[0], min(bounds[1], q_log))
            if target_type == "pct_change":
                raw = start_ktc * (1 + q_log)
            else:
                raw = start_ktc * np.exp(q_log)
            return max(KTC_MIN, min(KTC_MAX, raw))

        p20_model = q_dict.get(0.2)
        p80_model = q_dict.get(0.8)
        if p20_model is not None:
            p20_log = float(p20_model.predict(X)[0])
            result["p20_end_ktc"] = round(_bound_end_ktc(p20_log), 1)
        if p80_model is not None:
            p80_log = float(p80_model.predict(X)[0])
            result["p80_end_ktc"] = round(_bound_end_ktc(p80_log), 1)

    return result
