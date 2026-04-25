"""Precomputed prediction cache for all players.

Built lazily on first access per model. Takes ~30s but only happens once per model.
Uses the same build_prediction_inputs() as detail/what-if paths for consistency.
Cache is async — uses the app's event loop for DB access (no threading hacks).
"""

import asyncio
import logging
import os
import time

from app.services.data_loader import get_data_loader
from app.services.eos_model_service import build_prediction_inputs, predict_from_inputs
from app.services.sleeper import get_projections

logger = logging.getLogger(__name__)

CACHE_TTL_SECONDS = int(os.environ.get("KTC_CACHE_TTL", "900"))  # 15 min default

_caches: dict[str, dict[str, dict]] = {}
_built_at: dict[str, float] = {}
_building: set[str] = set()  # models currently being built (async lock)


async def _build(iteration) -> dict[str, dict]:
    """Compute predictions for all players using the shared input builder.

    Async so it can call get_latest_ktc_batch on the app's event loop.
    """
    dl = get_data_loader()
    players = dl.get_players()
    projected_ppg = get_projections()
    results = {}
    start = time.time()

    # Batch-fetch live KTC values on the running event loop (no threading)
    all_pids = [p["player_id"] for p in players]
    live_ktc_map: dict[str, float] = {}
    try:
        from app.services.ktc_db import get_latest_ktc_batch
        live_ktc_map = await get_latest_ktc_batch(all_pids)
    except Exception as e:
        logger.warning("Live KTC batch fetch failed (%s), using training data KTC", e)

    logger.info("Live KTC fetched for %d/%d players", len(live_ktc_map), len(all_pids))

    for player in players:
        pid = player["player_id"]
        live_ktc = live_ktc_map.get(pid)

        # Use the SAME builder as detail/what-if paths
        features = build_prediction_inputs(
            player, live_ktc=live_ktc, projected_ppg_map=projected_ppg,
        )
        if not features:
            continue

        gp = features["_baseline_games"]
        ppg = features["_baseline_ppg"]
        ppg_source = features["_ppg_source"]
        ktc_source = features["_ktc_source"]

        feat = {k: v for k, v in features.items() if not k.startswith("_")}

        try:
            pred = predict_from_inputs(
                iteration=iteration,
                games_played=gp,
                ppg=ppg,
                **feat,
            )
            results[pid] = {
                "predicted_end_ktc": round(pred["predicted_end_ktc"], 1),
                "predicted_delta_ktc": round(pred["predicted_delta_ktc"], 1),
                "predicted_pct_change": round(pred["predicted_pct_change"], 1),
                "ppg_used": round(ppg, 1),
                "ppg_source": ppg_source,
                "model_id": iteration.id,
                "start_ktc_used": round(features["start_ktc"], 1),
                "ktc_source": ktc_source,
            }
        except Exception as e:
            logger.debug("Cache prediction failed for %s (%s): %s", pid, player.get("name", "?"), e)

    elapsed = time.time() - start
    logger.info("Prediction cache built for model '%s': %d players in %.1fs", iteration.id, len(results), elapsed)
    return results


async def get_prediction_cache(model_id: str | None = None) -> dict[str, dict]:
    """Get the prediction cache for a model. Builds on first call per model (~30s).

    Rebuilds automatically after CACHE_TTL_SECONDS to pick up live KTC changes.
    Must be called from an async context (route handler or startup).
    """
    from app.services.model_registry import get_registry
    registry = get_registry()
    iteration = registry.get(model_id)
    mid = iteration.id

    now = time.time()
    built_time = _built_at.get(mid, 0)
    is_stale = (now - built_time) > CACHE_TTL_SECONDS if built_time else True

    if mid in _built_at and not is_stale:
        return _caches.get(mid, {})

    # Async guard: if another coroutine is already building, wait for it
    if mid in _building:
        while mid in _building:
            await asyncio.sleep(0.1)
        return _caches.get(mid, {})

    _building.add(mid)
    try:
        _caches[mid] = await _build(iteration)
    except Exception:
        logger.exception("Failed to build prediction cache for model '%s'", mid)
        if mid not in _caches:
            _caches[mid] = {}
    finally:
        _built_at[mid] = time.time()
        _building.discard(mid)

    return _caches.get(mid, {})
