"""EOS model service: wraps model registry for player-level predictions."""

import logging
from functools import lru_cache

import numpy as np

from app.services.ktc_utils import (
    _is_valid_ktc,
    compute_career_trajectory,
    compute_momentum_features,
    compute_prior_behavioral_features,
    compute_prior_ktc_features,
    compute_prior_position_stats,
    compute_prior_ppg,
    select_anchor_ktc,
    select_baseline_stats,
)
from app.services.model_registry import ModelIteration, get_registry
from app.services.ktc_db import get_latest_ktc

logger = logging.getLogger(__name__)


def _cap_ktc(x):
    if x is None:
        return None
    return max(1.0, min(9999.0, x))


# Tier-specific confidence band multipliers
_BAND_MULTIPLIERS = {
    "RB": {6000: 1.6, 4000: 1.2},
    "QB": {6000: 1.4, 4000: 1.15},
}


def _get_band_multiplier(position: str, start_ktc: float) -> float:
    if position not in _BAND_MULTIPLIERS:
        return 1.0
    thresholds = _BAND_MULTIPLIERS[position]
    for threshold in sorted(thresholds.keys(), reverse=True):
        if start_ktc >= threshold:
            return thresholds[threshold]
    return 1.0


def predict_from_inputs(
    iteration: ModelIteration,
    position: str,
    start_ktc: float,
    games_played: int,
    ppg: float,
    age: float | None = None,
    weeks_missed: float | None = None,
    draft_pick: float | None = None,
    years_remaining: float | None = None,
    start_position_rank: float | None = None,
    initial_ktc: float | None = None,
    min_ktc_prior: float | None = None,
    prior_end_ktc: float | None = None,
    max_ktc_prior: float | None = None,
    prior_ppg: float | None = None,
    # v3+ prior-behavioral signals (ignored by v1 via **_unused_kwargs)
    prior_weekly_fp_cv: float | None = None,
    prior_boom_rate: float | None = None,
    prior_bust_rate: float | None = None,
    prior_snap_pct: float | None = None,
    prior_ktc_volatility: float | None = None,
    # v4+ momentum + position stats (ignored by v1/v3 via **_unused_kwargs)
    ktc_30d_trend: float | None = None,
    ktc_90d_trend: float | None = None,
    momentum_ratio: float | None = None,
    max_games_missed_streak: float | None = None,
    prior_passing_tds: float | None = None,
    prior_interceptions: float | None = None,
    prior_carries: float | None = None,
    prior_red_zone_touches: float | None = None,
    prior_targets: float | None = None,
    prior_red_zone_targets: float | None = None,
    # v4+ efficiency features
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
    # v4+ career trajectory (all positions)
    prior_2yr_ppg: float | None = None,
    ppg_trend: float | None = None,
    prior_2yr_end_ktc: float | None = None,
    # v4+ team context (WR only; ignored by others via **_unused_kwargs)
    qb_ktc: float | None = None,
    team_total_ktc: float | None = None,
    positional_competition: float | None = None,
) -> dict:
    """Predict EOS KTC using a specific model iteration."""
    result = iteration.predict_end_ktc(
        position=position,
        gp=games_played,
        ppg=ppg,
        start_ktc=start_ktc,
        age=age,
        weeks_missed=weeks_missed,
        draft_pick=draft_pick,
        years_remaining=years_remaining,
        start_position_rank=start_position_rank,
        initial_ktc=initial_ktc,
        min_ktc_prior=min_ktc_prior,
        prior_end_ktc=prior_end_ktc,
        max_ktc_prior=max_ktc_prior,
        prior_ppg=prior_ppg,
        prior_weekly_fp_cv=prior_weekly_fp_cv,
        prior_boom_rate=prior_boom_rate,
        prior_bust_rate=prior_bust_rate,
        prior_snap_pct=prior_snap_pct,
        prior_ktc_volatility=prior_ktc_volatility,
        ktc_30d_trend=ktc_30d_trend,
        ktc_90d_trend=ktc_90d_trend,
        momentum_ratio=momentum_ratio,
        max_games_missed_streak=max_games_missed_streak,
        prior_passing_tds=prior_passing_tds,
        prior_interceptions=prior_interceptions,
        prior_carries=prior_carries,
        prior_red_zone_touches=prior_red_zone_touches,
        prior_targets=prior_targets,
        prior_red_zone_targets=prior_red_zone_targets,
        prior_completion_rate=prior_completion_rate,
        prior_rushing_yards=prior_rushing_yards,
        prior_pass_sacks=prior_pass_sacks,
        prior_yards_per_carry=prior_yards_per_carry,
        prior_receiving_yards=prior_receiving_yards,
        prior_rushing_tds=prior_rushing_tds,
        prior_yards_per_target=prior_yards_per_target,
        prior_air_yards_per_target=prior_air_yards_per_target,
        prior_receiving_tds=prior_receiving_tds,
        prior_drop_rate=prior_drop_rate,
        prior_2yr_ppg=prior_2yr_ppg,
        ppg_trend=ppg_trend,
        prior_2yr_end_ktc=prior_2yr_end_ktc,
        qb_ktc=qb_ktc,
        team_total_ktc=team_total_ktc,
        positional_competition=positional_competition,
    )

    effective_ktc = result.get("effective_start_ktc", start_ktc)

    # Confidence bands: prefer the per-player bands returned by the iteration
    # (computed via trained quantile models on the same feature vector).
    # Fall back to the static residual_bands percentile table for iterations
    # that don't return them (e.g. v1_hgb_baseline).
    low_end_ktc = None
    high_end_ktc = None
    if "p20_end_ktc" in result or "p80_end_ktc" in result:
        low_end_ktc = result.get("p20_end_ktc")
        high_end_ktc = result.get("p80_end_ktc")
    else:
        b = iteration.bundle
        bands = b.get("residual_bands", {}).get(position, {})
        if bands and effective_ktc > 0:
            pred_log = np.log(result["end_ktc"] / effective_ktc)
            multiplier = _get_band_multiplier(position, effective_ktc)
            low_end_ktc = round(effective_ktc * np.exp(pred_log + bands["p20"] * multiplier), 1)
            high_end_ktc = round(effective_ktc * np.exp(pred_log + bands["p80"] * multiplier), 1)

    predicted_end_ktc = _cap_ktc(result["end_ktc"])
    low_end_ktc = _cap_ktc(low_end_ktc)
    high_end_ktc = _cap_ktc(high_end_ktc)

    delta_ktc = predicted_end_ktc - effective_ktc if predicted_end_ktc else result["delta_ktc"]
    pct = (delta_ktc / effective_ktc * 100) if effective_ktc else 0.0

    return {
        "position": position,
        "start_ktc": round(effective_ktc, 1),
        "predicted_end_ktc": predicted_end_ktc,
        "predicted_delta_ktc": round(delta_ktc, 1),
        "predicted_pct_change": round(pct, 2),
        "low_end_ktc": low_end_ktc,
        "high_end_ktc": high_end_ktc,
        "model_version": iteration.id,
    }


async def predict_for_player(
    iteration: ModelIteration,
    player_id: str,
    data_loader,
) -> dict | None:
    """Predict EOS KTC for a player using a specific model iteration."""
    player = data_loader.get_player_by_id(player_id)
    if not player:
        return None

    seasons = player.get("seasons", [])
    if not seasons:
        return None

    # Prefer live KTC from database, fall back to training data
    live_ktc = None
    try:
        live_ktc = await get_latest_ktc(player_id)
    except Exception:
        pass

    anchor = select_anchor_ktc(seasons)
    if live_ktc and live_ktc > 0:
        start_ktc = live_ktc
        anchor_year = anchor[1] if anchor else max(s["year"] for s in seasons)
        anchor_source = "live_db"
    elif anchor:
        start_ktc, anchor_year, anchor_source = anchor
    else:
        return None

    # Baseline stats
    latest = max(seasons, key=lambda s: s["year"])
    baseline_info = select_baseline_stats(seasons)
    if baseline_info:
        baseline_year, games, ppg = baseline_info
    else:
        baseline_year = latest["year"]
        games = 0
        ppg = 0.0

    baseline_season = next(
        (s for s in seasons if s["year"] == baseline_year), latest
    )
    age = baseline_season.get("age") or latest.get("age")

    # Prior-season features (all positions — model trains with these for everyone)
    prior_end_ktc = None
    max_ktc_prior = None
    prior_ppg = None
    prior_ref_year = anchor_year if anchor_year else (latest["year"] + 1)
    prior_end_ktc, max_ktc_prior, initial_ktc, min_ktc_prior = compute_prior_ktc_features(seasons, prior_ref_year)
    prior_ppg = compute_prior_ppg(seasons, prior_ref_year)

    # Prior-season behavioral signals (v3+). Computed for all positions;
    # older iterations receive them as kwargs and ignore via **_unused_kwargs.
    prior_behavioral = compute_prior_behavioral_features(seasons, prior_ref_year) or {}

    # v4+ momentum features (from current/anchor season)
    momentum = compute_momentum_features(seasons, anchor_year) or {}

    # v4+ position-specific prior stats
    prior_pos = compute_prior_position_stats(seasons, player["position"], prior_ref_year) or {}

    # Career trajectory (2-year lookback)
    trajectory = compute_career_trajectory(seasons, prior_ref_year) or {}

    start_position_rank = latest.get("start_position_rank")

    result = predict_from_inputs(
        iteration=iteration,
        position=player["position"],
        start_ktc=start_ktc,
        games_played=games,
        ppg=ppg,
        age=float(age) if age is not None else None,
        start_position_rank=float(start_position_rank) if start_position_rank is not None else None,
        initial_ktc=initial_ktc,
        min_ktc_prior=min_ktc_prior,
        prior_end_ktc=prior_end_ktc,
        max_ktc_prior=max_ktc_prior,
        prior_ppg=prior_ppg,
        prior_weekly_fp_cv=prior_behavioral.get("prior_weekly_fp_cv"),
        prior_boom_rate=prior_behavioral.get("prior_boom_rate"),
        prior_bust_rate=prior_behavioral.get("prior_bust_rate"),
        prior_snap_pct=prior_behavioral.get("prior_snap_pct"),
        prior_ktc_volatility=prior_behavioral.get("prior_ktc_volatility"),
        ktc_30d_trend=momentum.get("ktc_30d_trend"),
        ktc_90d_trend=momentum.get("ktc_90d_trend"),
        momentum_ratio=momentum.get("momentum_ratio"),
        max_games_missed_streak=momentum.get("max_games_missed_streak"),
        prior_passing_tds=prior_pos.get("prior_passing_tds"),
        prior_interceptions=prior_pos.get("prior_interceptions"),
        prior_carries=prior_pos.get("prior_carries"),
        prior_red_zone_touches=prior_pos.get("prior_red_zone_touches"),
        prior_targets=prior_pos.get("prior_targets"),
        prior_red_zone_targets=prior_pos.get("prior_red_zone_targets"),
        prior_completion_rate=prior_pos.get("prior_completion_rate"),
        prior_rushing_yards=prior_pos.get("prior_rushing_yards"),
        prior_pass_sacks=prior_pos.get("prior_pass_sacks"),
        prior_yards_per_carry=prior_pos.get("prior_yards_per_carry"),
        prior_receiving_yards=prior_pos.get("prior_receiving_yards"),
        prior_rushing_tds=prior_pos.get("prior_rushing_tds"),
        prior_yards_per_target=prior_pos.get("prior_yards_per_target"),
        prior_air_yards_per_target=prior_pos.get("prior_air_yards_per_target"),
        prior_receiving_tds=prior_pos.get("prior_receiving_tds"),
        prior_drop_rate=prior_pos.get("prior_drop_rate"),
        # Career trajectory
        prior_2yr_ppg=trajectory.get("prior_2yr_ppg"),
        ppg_trend=trajectory.get("ppg_trend"),
        prior_2yr_end_ktc=trajectory.get("prior_2yr_end_ktc"),
        # Team context features: pulled from latest season training data.
        qb_ktc=latest.get("qb_ktc") if latest else None,
        team_total_ktc=latest.get("team_total_ktc") if latest else None,
        positional_competition=latest.get("positional_competition") if latest else None,
    )
    result["player_id"] = player_id
    result["name"] = player["name"]
    result["anchor_year"] = anchor_year
    result["anchor_source"] = anchor_source
    result["baseline_year"] = baseline_year
    return result


async def predict_for_player_whatif(
    iteration: ModelIteration,
    player_id: str,
    data_loader,
    games_played: int,
    ppg: float,
) -> dict | None:
    """Like predict_for_player but with overridden games_played and ppg.

    Uses all the player's real context (prior stats, momentum, team, etc.)
    but lets the caller control the scenario inputs.
    """
    player = data_loader.get_player_by_id(player_id)
    if not player:
        return None

    seasons = player.get("seasons", [])
    if not seasons:
        return None

    live_ktc = None
    try:
        live_ktc = await get_latest_ktc(player_id)
    except Exception:
        pass

    anchor = select_anchor_ktc(seasons)
    if live_ktc and live_ktc > 0:
        start_ktc = live_ktc
        anchor_year = anchor[1] if anchor else max(s["year"] for s in seasons)
    elif anchor:
        start_ktc, anchor_year, _ = anchor
    else:
        return None

    latest = max(seasons, key=lambda s: s["year"])
    baseline_info = select_baseline_stats(seasons)
    baseline_year = baseline_info[0] if baseline_info else latest["year"]
    baseline_season = next(
        (s for s in seasons if s["year"] == baseline_year), latest
    )
    age = baseline_season.get("age") or latest.get("age")

    prior_ref_year = anchor_year if anchor_year else (latest["year"] + 1)
    prior_end_ktc, max_ktc_prior, initial_ktc, min_ktc_prior = compute_prior_ktc_features(seasons, prior_ref_year)
    prior_ppg = compute_prior_ppg(seasons, prior_ref_year)
    prior_behavioral = compute_prior_behavioral_features(seasons, prior_ref_year) or {}
    momentum = compute_momentum_features(seasons, anchor_year) or {}
    prior_pos = compute_prior_position_stats(seasons, player["position"], prior_ref_year) or {}
    trajectory = compute_career_trajectory(seasons, prior_ref_year) or {}
    start_position_rank = latest.get("start_position_rank")

    result = predict_from_inputs(
        iteration=iteration,
        position=player["position"],
        start_ktc=start_ktc,
        games_played=games_played,
        ppg=ppg,
        age=float(age) if age is not None else None,
        start_position_rank=float(start_position_rank) if start_position_rank is not None else None,
        initial_ktc=initial_ktc,
        min_ktc_prior=min_ktc_prior,
        prior_end_ktc=prior_end_ktc,
        max_ktc_prior=max_ktc_prior,
        prior_ppg=prior_ppg,
        prior_weekly_fp_cv=prior_behavioral.get("prior_weekly_fp_cv"),
        prior_boom_rate=prior_behavioral.get("prior_boom_rate"),
        prior_bust_rate=prior_behavioral.get("prior_bust_rate"),
        prior_snap_pct=prior_behavioral.get("prior_snap_pct"),
        prior_ktc_volatility=prior_behavioral.get("prior_ktc_volatility"),
        ktc_30d_trend=momentum.get("ktc_30d_trend"),
        ktc_90d_trend=momentum.get("ktc_90d_trend"),
        momentum_ratio=momentum.get("momentum_ratio"),
        max_games_missed_streak=momentum.get("max_games_missed_streak"),
        prior_passing_tds=prior_pos.get("prior_passing_tds"),
        prior_interceptions=prior_pos.get("prior_interceptions"),
        prior_carries=prior_pos.get("prior_carries"),
        prior_red_zone_touches=prior_pos.get("prior_red_zone_touches"),
        prior_targets=prior_pos.get("prior_targets"),
        prior_red_zone_targets=prior_pos.get("prior_red_zone_targets"),
        prior_completion_rate=prior_pos.get("prior_completion_rate"),
        prior_rushing_yards=prior_pos.get("prior_rushing_yards"),
        prior_pass_sacks=prior_pos.get("prior_pass_sacks"),
        prior_yards_per_carry=prior_pos.get("prior_yards_per_carry"),
        prior_receiving_yards=prior_pos.get("prior_receiving_yards"),
        prior_rushing_tds=prior_pos.get("prior_rushing_tds"),
        prior_yards_per_target=prior_pos.get("prior_yards_per_target"),
        prior_air_yards_per_target=prior_pos.get("prior_air_yards_per_target"),
        prior_receiving_tds=prior_pos.get("prior_receiving_tds"),
        prior_drop_rate=prior_pos.get("prior_drop_rate"),
        prior_2yr_ppg=trajectory.get("prior_2yr_ppg"),
        ppg_trend=trajectory.get("ppg_trend"),
        prior_2yr_end_ktc=trajectory.get("prior_2yr_end_ktc"),
        qb_ktc=latest.get("qb_ktc") if latest else None,
        team_total_ktc=latest.get("team_total_ktc") if latest else None,
        positional_competition=latest.get("positional_competition") if latest else None,
    )
    return result


def predict_historical(
    player_id: str,
    data_loader,
) -> list[dict]:
    """Predict EOS KTC for a player across all available temporal models.

    For each temporal iteration (v3_for_2021, v3_for_2022, etc.), uses
    the player's actual start_ktc and stats for that year's season,
    then compares against the actual end_ktc.

    Returns a list of dicts sorted by year, each with the prediction
    and actual outcome for that season.
    """
    registry = get_registry()
    player = data_loader.get_player_by_id(player_id)
    if not player:
        return []

    seasons = player.get("seasons", [])
    seasons_by_year = {s["year"]: s for s in seasons}

    results = []

    for it_info in registry.list_iterations():
        if not it_info.get("temporal"):
            continue

        predict_year = it_info.get("predict_year")
        if not predict_year or predict_year not in seasons_by_year:
            continue

        season = seasons_by_year[predict_year]
        start_ktc = season.get("start_ktc")
        end_ktc = season.get("end_ktc")
        games = season.get("games_played", 0) or 0
        fp = season.get("fantasy_points", 0) or 0
        ppg = fp / games if games > 0 else 0.0
        age = season.get("age")

        if not start_ktc or start_ktc <= 0:
            continue

        try:
            iteration = registry.get(it_info["id"])
        except KeyError:
            continue

        # Prior-season features relative to predict_year (all positions)
        prior_end_ktc, max_ktc_prior, _, _ = compute_prior_ktc_features(seasons, predict_year)
        prior_ppg_val = compute_prior_ppg(seasons, predict_year)

        prior_behavioral = compute_prior_behavioral_features(seasons, predict_year) or {}

        try:
            pred = predict_from_inputs(
                iteration=iteration,
                position=player["position"],
                start_ktc=start_ktc,
                games_played=games,
                ppg=ppg,
                age=float(age) if age is not None else None,
                prior_end_ktc=prior_end_ktc,
                max_ktc_prior=max_ktc_prior,
                prior_ppg=prior_ppg_val,
                prior_weekly_fp_cv=prior_behavioral.get("prior_weekly_fp_cv"),
                prior_boom_rate=prior_behavioral.get("prior_boom_rate"),
                prior_bust_rate=prior_behavioral.get("prior_bust_rate"),
                prior_snap_pct=prior_behavioral.get("prior_snap_pct"),
                prior_ktc_volatility=prior_behavioral.get("prior_ktc_volatility"),
            )
        except Exception:
            continue

        actual_end_ktc = end_ktc if end_ktc and 0 < end_ktc < 9999 else None
        error = None
        if actual_end_ktc is not None:
            error = round(pred["predicted_end_ktc"] - actual_end_ktc, 1)

        results.append({
            "year": predict_year,
            "model_version": it_info["id"],
            "start_ktc": round(start_ktc, 1),
            "actual_end_ktc": round(actual_end_ktc, 1) if actual_end_ktc else None,
            "predicted_end_ktc": pred["predicted_end_ktc"],
            "predicted_delta_ktc": pred["predicted_delta_ktc"],
            "predicted_pct_change": pred["predicted_pct_change"],
            "low_end_ktc": pred.get("low_end_ktc"),
            "high_end_ktc": pred.get("high_end_ktc"),
            "error": error,
            "games_played": games,
            "ppg": round(ppg, 1),
        })

    results.sort(key=lambda r: r["year"])
    return results
