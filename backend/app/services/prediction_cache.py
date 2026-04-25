"""Precomputed prediction cache for all players.

Built lazily on first access per model. Takes ~30s but only happens once per model.
Uses the same build_prediction_inputs() as detail/what-if paths for consistency.
"""

import asyncio
import logging
import threading
import time

from app.services.data_loader import get_data_loader
from app.services.eos_model_service import build_prediction_inputs, predict_from_inputs
from app.services.sleeper import get_projections

logger = logging.getLogger(__name__)

_caches: dict[str, dict[str, dict]] = {}
_built_models: set[str] = set()
_lock = threading.Lock()


def _fetch_live_ktc_batch(player_ids: list[str]) -> dict[str, float]:
    """Fetch live KTC values from the DB (sync wrapper for async batch call).

    Falls back gracefully to empty dict if DB is unavailable.
    """
    import os
    if not os.getenv("DATABASE_URL"):
        return {}

    from app.services.ktc_db import get_latest_ktc_batch
    try:
        import concurrent.futures
        with concurrent.futures.ThreadPoolExecutor() as pool:
            return pool.submit(
                lambda: asyncio.run(get_latest_ktc_batch(player_ids))
            ).result(timeout=30)
    except Exception as e:
        logger.warning("Live KTC batch fetch failed (%s), using training data KTC", e)
        return {}


def _build(iteration) -> dict[str, dict]:
    """Compute predictions for all players using the shared input builder."""
    dl = get_data_loader()
    players = dl.get_players()
    projected_ppg = get_projections()
    results = {}
    start = time.time()

    # Batch-fetch live KTC values (same source as /api/players display)
    all_pids = [p["player_id"] for p in players]
    live_ktc_map = _fetch_live_ktc_batch(all_pids)
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
                "projected_ppg": round(ppg, 1),
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


def get_prediction_cache(model_id: str | None = None) -> dict[str, dict]:
    """Get the prediction cache for a model. Builds on first call per model (~30s)."""
    from app.services.model_registry import get_registry
    registry = get_registry()
    iteration = registry.get(model_id)
    mid = iteration.id

    if mid in _built_models:
        return _caches.get(mid, {})

    with _lock:
        if mid not in _built_models:
            try:
                _caches[mid] = _build(iteration)
            except Exception:
                logger.exception("Failed to build prediction cache for model '%s'", mid)
                _caches[mid] = {}
            _built_models.add(mid)

    return _caches.get(mid, {})
