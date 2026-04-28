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
    # Sleeper preseason projected PPG
    projected_ppg: float | None = None,
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
        projected_ppg=projected_ppg,
        has_projected_ppg=1 if projected_ppg is not None else 0,
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
    features = await _compute_player_features(player_id, data_loader)
    if not features:
        return None

    player = features["_player"]
    anchor_year = features["_anchor_year"]
    baseline_year = features["_baseline_year"]
    games = features["_baseline_games"]
    ppg = features["_baseline_ppg"]
    ppg_source = features.get("_ppg_source", "last_season")
    ktc_source = features.get("_ktc_source", "unknown")

    feat = {k: v for k, v in features.items() if not k.startswith("_")}

    result = predict_from_inputs(
        iteration=iteration,
        games_played=games,
        ppg=ppg,
        **feat,
    )
    result["player_id"] = player_id
    result["name"] = player["name"]
    result["anchor_year"] = anchor_year
    result["anchor_source"] = ktc_source
    result["baseline_year"] = baseline_year
    result["prediction_meta"] = {
        "model_id": iteration.id,
        "start_ktc_used": result["start_ktc"],
        "ppg_used": round(ppg, 1),
        "ppg_source": ppg_source,
        "ktc_source": ktc_source,
    }
    return result


def build_prediction_inputs(
    player: dict,
    live_ktc: float | None = None,
    projected_ppg_map: dict[str, float] | None = None,
) -> dict | None:
    """Shared sync builder for prediction inputs. Used by cache, detail, and what-if paths.

    Returns a kwargs dict for predict_from_inputs with metadata in underscore-prefixed keys,
    or None if insufficient data.

    Parameters
    ----------
    player : dict
        Player dict with seasons, player_id, name, position.
    live_ktc : float | None
        Pre-fetched live KTC value from DB. If None, falls back to training data.
    projected_ppg_map : dict | None
        Sleeper projections map. If None, fetches via get_projections().
    """
    player_id = player["player_id"]
    seasons = player.get("seasons", [])
    if not seasons:
        return None

    if projected_ppg_map is None:
        from app.services.sleeper import get_projections
        projected_ppg_map = get_projections()

    anchor = select_anchor_ktc(seasons)
    if live_ktc and live_ktc > 0:
        start_ktc = live_ktc
        anchor_year = anchor[1] if anchor else max(s["year"] for s in seasons)
        ktc_source = "live_db"
    elif anchor:
        start_ktc, anchor_year, _ = anchor
        ktc_source = "training_data"
    else:
        return None

    latest = max(seasons, key=lambda s: s["year"])
    baseline_info = select_baseline_stats(seasons)
    baseline_year = baseline_info[0] if baseline_info else latest["year"]
    baseline_season = next(
        (s for s in seasons if s["year"] == baseline_year), latest
    )
    age = baseline_season.get("age") or latest.get("age")

    # Adjust age for current year if training data is behind
    from datetime import date
    year_gap = date.today().year - latest["year"]
    if year_gap > 0 and age is not None:
        age = age + year_gap

    # Unify PPG source: prefer Sleeper projections
    baseline_gp = baseline_info[1] if baseline_info else 0
    baseline_ppg = baseline_info[2] if baseline_info else 0.0
    proj_ppg = projected_ppg_map.get(player_id)
    if proj_ppg is not None:
        ppg_used = proj_ppg
        gp_used = 17
        ppg_source = "projected"
    else:
        ppg_used = baseline_ppg
        gp_used = baseline_gp
        ppg_source = "last_season"

    prior_ref_year = anchor_year if anchor_year else (latest["year"] + 1)
    prior_end_ktc, max_ktc_prior, initial_ktc, min_ktc_prior = compute_prior_ktc_features(seasons, prior_ref_year)
    prior_ppg_val = compute_prior_ppg(seasons, prior_ref_year)
    prior_behavioral = compute_prior_behavioral_features(seasons, prior_ref_year) or {}
    momentum = compute_momentum_features(seasons, anchor_year) or {}
    prior_pos = compute_prior_position_stats(seasons, player["position"], prior_ref_year) or {}
    trajectory = compute_career_trajectory(seasons, prior_ref_year) or {}
    start_position_rank = latest.get("start_position_rank")

    # Fallback for year-1/2 players with no qualifying prior season:
    # use the current season as "prior" so the model has something to work with.
    # Without this, all prior features are NaN and prediction lines go flat.
    if prior_ppg_val is None and baseline_gp >= 4:
        prior_ppg_val = baseline_ppg
    if not prior_behavioral and baseline_gp >= 4:
        prior_behavioral = {
            "prior_weekly_fp_cv": float(baseline_season.get("weekly_fp_cv") or 0.0),
            "prior_boom_rate": float(baseline_season.get("boom_rate") or 0.0),
            "prior_bust_rate": float(baseline_season.get("bust_rate") or 0.0),
            "prior_snap_pct": float(baseline_season.get("snap_pct") or 0.0),
            "prior_ktc_volatility": float(baseline_season.get("ktc_volatility") or 0.0),
        }
    if not prior_pos and baseline_gp >= 4:
        prior_pos = compute_prior_position_stats(
            seasons, player["position"], prior_ref_year, min_games=0
        ) or {}
        # If still nothing from prior seasons, use the current/baseline season directly
        if not prior_pos:
            pos = player["position"]
            s = baseline_season
            if pos == "RB":
                carries = float(s.get("carries") or 0)
                rush_yds = float(s.get("rushing_yards") or 0)
                prior_pos = {
                    "prior_carries": carries,
                    "prior_red_zone_touches": float(s.get("red_zone_touches") or 0),
                    "prior_yards_per_carry": rush_yds / carries if carries > 0 else 0.0,
                    "prior_receiving_yards": float(s.get("receiving_yards") or 0),
                    "prior_rushing_tds": float(s.get("rushing_tds") or 0),
                }
            elif pos == "WR":
                prior_pos = {
                    "prior_targets": float(s.get("targets") or 0),
                    "prior_red_zone_targets": float(s.get("red_zone_targets") or 0),
                    "prior_yards_per_target": float(s.get("yards_per_target") or 0),
                    "prior_air_yards_per_target": float(s.get("air_yards_per_target") or 0),
                    "prior_receiving_tds": float(s.get("receiving_tds") or 0),
                    "prior_drop_rate": float(s.get("drop_rate") or 0),
                }
            elif pos == "TE":
                prior_pos = {
                    "prior_targets": float(s.get("targets") or 0),
                    "prior_red_zone_targets": float(s.get("red_zone_targets") or 0),
                    "prior_yards_per_target": float(s.get("yards_per_target") or 0),
                    "prior_receiving_tds": float(s.get("receiving_tds") or 0),
                    "prior_drop_rate": float(s.get("drop_rate") or 0),
                }
            elif pos == "QB":
                prior_pos = {
                    "prior_passing_tds": float(s.get("passing_tds") or 0),
                    "prior_interceptions": float(s.get("interceptions") or 0),
                    "prior_completion_rate": float(s.get("completion_rate") or 0),
                    "prior_rushing_yards": float(s.get("rushing_yards") or 0),
                    "prior_pass_sacks": float(s.get("pass_sacks") or 0),
                }

    return {
        "position": player["position"],
        "start_ktc": start_ktc,
        "age": float(age) if age is not None else None,
        "start_position_rank": float(start_position_rank) if start_position_rank is not None else None,
        "initial_ktc": initial_ktc,
        "min_ktc_prior": min_ktc_prior,
        "prior_end_ktc": prior_end_ktc,
        "max_ktc_prior": max_ktc_prior,
        "prior_ppg": prior_ppg_val,
        "prior_weekly_fp_cv": prior_behavioral.get("prior_weekly_fp_cv"),
        "prior_boom_rate": prior_behavioral.get("prior_boom_rate"),
        "prior_bust_rate": prior_behavioral.get("prior_bust_rate"),
        "prior_snap_pct": prior_behavioral.get("prior_snap_pct"),
        "prior_ktc_volatility": prior_behavioral.get("prior_ktc_volatility"),
        "ktc_30d_trend": momentum.get("ktc_30d_trend"),
        "ktc_90d_trend": momentum.get("ktc_90d_trend"),
        "momentum_ratio": momentum.get("momentum_ratio"),
        "max_games_missed_streak": momentum.get("max_games_missed_streak"),
        "prior_passing_tds": prior_pos.get("prior_passing_tds"),
        "prior_interceptions": prior_pos.get("prior_interceptions"),
        "prior_carries": prior_pos.get("prior_carries"),
        "prior_red_zone_touches": prior_pos.get("prior_red_zone_touches"),
        "prior_targets": prior_pos.get("prior_targets"),
        "prior_red_zone_targets": prior_pos.get("prior_red_zone_targets"),
        "prior_completion_rate": prior_pos.get("prior_completion_rate"),
        "prior_rushing_yards": prior_pos.get("prior_rushing_yards"),
        "prior_pass_sacks": prior_pos.get("prior_pass_sacks"),
        "prior_yards_per_carry": prior_pos.get("prior_yards_per_carry"),
        "prior_receiving_yards": prior_pos.get("prior_receiving_yards"),
        "prior_rushing_tds": prior_pos.get("prior_rushing_tds"),
        "prior_yards_per_target": prior_pos.get("prior_yards_per_target"),
        "prior_air_yards_per_target": prior_pos.get("prior_air_yards_per_target"),
        "prior_receiving_tds": prior_pos.get("prior_receiving_tds"),
        "prior_drop_rate": prior_pos.get("prior_drop_rate"),
        "prior_2yr_ppg": trajectory.get("prior_2yr_ppg"),
        "ppg_trend": trajectory.get("ppg_trend"),
        "prior_2yr_end_ktc": trajectory.get("prior_2yr_end_ktc"),
        "qb_ktc": latest.get("qb_ktc") if latest else None,
        "team_total_ktc": latest.get("team_total_ktc") if latest else None,
        "positional_competition": latest.get("positional_competition") if latest else None,
        "projected_ppg": proj_ppg,
        # Extra context for caller
        "_player": player,
        "_anchor_year": anchor_year,
        "_baseline_year": baseline_year,
        "_baseline_games": gp_used,
        "_baseline_ppg": ppg_used,
        "_ppg_source": ppg_source,
        "_ktc_source": ktc_source,
    }


async def _compute_player_features(player_id: str, data_loader) -> dict | None:
    """Async wrapper around build_prediction_inputs that fetches live KTC from DB."""
    player = data_loader.get_player_by_id(player_id)
    if not player:
        return None

    live_ktc = None
    try:
        live_ktc = await get_latest_ktc(player_id)
    except Exception:
        logger.debug("Live KTC unavailable for %s", player_id)

    return build_prediction_inputs(player, live_ktc=live_ktc)


async def predict_for_player_whatif(
    iteration: ModelIteration,
    player_id: str,
    data_loader,
    games_played: int,
    ppg: float,
    _features: dict | None = None,
) -> dict | None:
    """Like predict_for_player but with overridden games_played and ppg.

    Accepts pre-computed _features dict to avoid redundant computation
    when called in a batch loop.
    """
    features = _features or await _compute_player_features(player_id, data_loader)
    if not features:
        return None

    # Extract and remove internal keys
    player = features.get("_player", {})
    feat = {k: v for k, v in features.items() if not k.startswith("_")}

    result = predict_from_inputs(
        iteration=iteration,
        games_played=games_played,
        ppg=ppg,
        **feat,
    )
    result["player_id"] = player_id
    result["name"] = player.get("name")
    result["prediction_meta"] = {
        "model_id": iteration.id,
        "start_ktc_used": result["start_ktc"],
        "ppg_used": round(ppg, 1),
        "ppg_source": "manual_override",
        "ktc_source": features.get("_ktc_source"),
    }
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
