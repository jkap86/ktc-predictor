"""v5 short-window tool: what 3 or 6 games do to a player's dynasty value."""

import asyncio
from functools import lru_cache

from fastapi import APIRouter, HTTPException, Query

from predictor_v5.window_tool import WindowTool

router = APIRouter(prefix="/api/v5/window", tags=["window"])

MAX_POINTS = 250.0
MAX_SWEEP = 100


@lru_cache(maxsize=4)
def get_tool(season: int) -> WindowTool:
    tool = WindowTool(season=season)
    _ = tool.inputs  # fail fast if the inputs file is missing
    return tool


def _tool(season: int) -> WindowTool:
    try:
        return get_tool(season)
    except (FileNotFoundError, KeyError) as e:
        raise HTTPException(status_code=404, detail=f"No window inputs for {season}: {e}")


@router.get("")
def window_info(season: int = Query(2026)):
    return _tool(season).info()


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
    games: int = Query(3, description="window length: 3 or 6 games"),
    start_week: int | None = Query(None, ge=1, le=18, description="first week of the window; the latest finished one if omitted"),
    points: str | None = Query(None, description="window points, or a comma-separated list; the actual points if omitted"),
    window_games: float | None = Query(None, ge=0, le=6, description="games played in the window"),
    season: int = Query(2026),
):
    values = None
    if points:
        try:
            values = [float(p) for p in points.split(",") if p.strip()]
        except ValueError:
            raise HTTPException(status_code=400, detail="points must be numbers")
        if not values or len(values) > MAX_SWEEP or any(not 0 <= v <= MAX_POINTS for v in values):
            raise HTTPException(status_code=400, detail=f"1-{MAX_SWEEP} points values between 0 and {MAX_POINTS:g}")
    tool = _tool(season)
    try:
        # The first call per source and length trains a model (seconds); keep it off the event loop.
        return await asyncio.to_thread(tool.predict, player_id, games, start_week, values, window_games)
    except KeyError as e:
        raise HTTPException(status_code=404, detail=str(e))
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
