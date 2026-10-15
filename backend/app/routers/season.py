"""v5 season tool: pre-season value + projected season points -> end-of-season value."""

import asyncio
from functools import lru_cache

from fastapi import APIRouter, HTTPException, Query

from predictor_v5.season import SOURCES, SeasonTool

router = APIRouter(prefix="/api/v5/season", tags=["season"])

MAX_POINTS = 600.0
MAX_SWEEP = 100


@lru_cache(maxsize=4)
def get_tool(season: int) -> SeasonTool:
    tool = SeasonTool(season=season)
    _ = tool.players  # fail fast if the start values file is missing
    return tool


def _tool(season: int) -> SeasonTool:
    try:
        return get_tool(season)
    except (FileNotFoundError, KeyError) as e:
        raise HTTPException(status_code=404, detail=f"No start values for {season}: {e}")


@router.get("")
def season_info(season: int = Query(2026)):
    tool = _tool(season)
    return {
        "season": tool.season,
        "cutoff": tool.cutoff.isoformat(),
        "players": len(tool.players),
        "sources": {name: cfg["label"] for name, cfg in SOURCES.items()},
    }


@router.get("/players")
def search_players(
    q: str = Query("", max_length=100),
    position: str | None = Query(None, pattern="^(QB|RB|WR|TE)$"),
    limit: int = Query(25, ge=1, le=200),
    season: int = Query(2026),
):
    return {"players": _tool(season).search(q, position, limit)}


@router.get("/predict")
async def predict(
    player_id: str = Query(..., max_length=32),
    points: str = Query(..., description="season points, or a comma-separated list"),
    age: float | None = Query(None, ge=18, le=45),
    season: int = Query(2026),
):
    try:
        values = [float(p) for p in points.split(",") if p.strip()]
    except ValueError:
        raise HTTPException(status_code=400, detail="points must be numbers")
    if not values or len(values) > MAX_SWEEP or any(not 0 <= v <= MAX_POINTS for v in values):
        raise HTTPException(status_code=400, detail=f"1-{MAX_SWEEP} points values between 0 and {MAX_POINTS:g}")
    tool = _tool(season)
    try:
        # The first call trains the models (a few seconds); keep it off the event loop.
        return await asyncio.to_thread(tool.predict, player_id, values, age)
    except KeyError as e:
        raise HTTPException(status_code=404, detail=str(e))
