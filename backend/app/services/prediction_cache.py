"""Precomputed prediction cache for all players.

Built lazily on first access per model. Takes ~30s but only happens once per model.
Uses Sleeper projections for default PPG when available.
"""

import logging
import threading
import time

from app.services.data_loader import get_data_loader
from app.services.ktc_utils import (
    select_anchor_ktc,
    select_baseline_stats,
    compute_prior_ktc_features,
    compute_prior_ppg,
    compute_prior_behavioral_features,
    compute_momentum_features,
    compute_prior_position_stats,
    compute_career_trajectory,
)
from app.services.eos_model_service import predict_from_inputs
from app.services.sleeper import get_projections

logger = logging.getLogger(__name__)

_caches: dict[str, dict[str, dict]] = {}
_built_models: set[str] = set()
_lock = threading.Lock()


def _build(iteration) -> dict[str, dict]:
    """Compute predictions for all players using a specific model iteration."""
    dl = get_data_loader()
    players = dl.get_players()
    projected_ppg = get_projections()
    results = {}
    start = time.time()

    for player in players:
        pid = player["player_id"]
        seasons = player.get("seasons", [])
        if not seasons:
            continue

        anchor = select_anchor_ktc(seasons)
        if not anchor:
            continue
        start_ktc, anchor_year, _ = anchor

        baseline = select_baseline_stats(seasons)
        if baseline:
            _, gp, ppg = baseline
        else:
            gp, ppg = 0, 0.0

        # Use Sleeper projected PPG if available (better than last-season PPG)
        proj_ppg = projected_ppg.get(pid)
        ppg_source = "last_season"
        if proj_ppg is not None:
            ppg = proj_ppg
            gp = 17  # projected for full season
            ppg_source = "projected"

        latest = max(seasons, key=lambda s: s["year"])
        age = latest.get("age") or 25

        prior_ref_year = anchor_year or (latest["year"] + 1)
        prior_end_ktc, max_ktc_prior, initial_ktc, min_ktc_prior = compute_prior_ktc_features(seasons, prior_ref_year)
        prior_ppg = compute_prior_ppg(seasons, prior_ref_year)
        prior_behavioral = compute_prior_behavioral_features(seasons, prior_ref_year) or {}
        momentum = compute_momentum_features(seasons, anchor_year) or {}
        prior_pos = compute_prior_position_stats(seasons, player["position"], prior_ref_year) or {}
        trajectory = compute_career_trajectory(seasons, prior_ref_year) or {}

        try:
            pred = predict_from_inputs(
                iteration=iteration,
                position=player["position"],
                start_ktc=start_ktc,
                games_played=gp,
                ppg=ppg,
                age=float(age),
                start_position_rank=float(latest.get("start_position_rank")) if latest.get("start_position_rank") is not None else None,
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
                qb_ktc=latest.get("qb_ktc"),
                team_total_ktc=latest.get("team_total_ktc"),
                positional_competition=latest.get("positional_competition"),
                projected_ppg=proj_ppg,
            )
            results[pid] = {
                "predicted_end_ktc": round(pred["predicted_end_ktc"], 1),
                "predicted_delta_ktc": round(pred["predicted_delta_ktc"], 1),
                "predicted_pct_change": round(pred["predicted_pct_change"], 1),
                "projected_ppg": round(ppg, 1),
                "ppg_source": ppg_source,
                "model_id": iteration.id,
            }
        except Exception as e:
            logger.debug("Cache prediction failed for %s (%s): %s", pid, player.get("name", "?"), e)

    elapsed = time.time() - start
    logger.info("Prediction cache built for model '%s': %d players in %.1fs", iteration.id, len(results), elapsed)
    return results


def get_prediction_cache(model_id: str | None = None) -> dict[str, dict]:
    """Get the prediction cache for a model. Builds on first call per model (~30s)."""
    from app.services.model_registry import get_registry
    registry = get_registry()
    iteration = registry.get(model_id)
    mid = iteration.id

    if mid in _built_models:
        return _caches.get(mid, {})

    with _lock:
        # Double-check after acquiring lock
        if mid not in _built_models:
            try:
                _caches[mid] = _build(iteration)
            except Exception:
                logger.exception("Failed to build prediction cache for model '%s'", mid)
                _caches[mid] = {}
            _built_models.add(mid)

    return _caches.get(mid, {})
