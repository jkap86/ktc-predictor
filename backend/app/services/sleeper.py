"""Sleeper API integration for player projections."""

import json
import logging
import urllib.request
from functools import lru_cache

from app.config import PROJECTION_SEASON

logger = logging.getLogger(__name__)


def fetch_projections(season: int = PROJECTION_SEASON) -> dict[str, float]:
    """Fetch projected PPG from Sleeper API. Returns {player_id: ppg}."""
    try:
        url = f"https://api.sleeper.com/projections/nfl/{season}/?season_type=regular"
        req = urllib.request.Request(url, headers={"User-Agent": "KTC-Predictor/1.0"})
        with urllib.request.urlopen(req, timeout=15) as resp:
            data = json.loads(resp.read())
        result = {}
        for item in data:
            pid = str(item.get("player_id", ""))
            stats = item.get("stats", {})
            pts = stats.get("pts_half_ppr")
            gp = stats.get("gp")
            if pts and gp and gp > 0:
                result[pid] = round(pts / gp, 2)
        logger.info("Fetched %d projected PPG values from Sleeper (season %d)", len(result), season)
        return result
    except Exception:
        logger.warning("Failed to fetch Sleeper projections for season %d", season)
        return {}


@lru_cache(maxsize=1)
def get_projections() -> dict[str, float]:
    """Cached Sleeper projections — fetched once per process."""
    return fetch_projections()
