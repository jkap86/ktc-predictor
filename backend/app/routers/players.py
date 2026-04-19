"""Player listing and detail endpoints."""

import logging
from typing import Optional

from fastapi import APIRouter, HTTPException, Query

from app.services.data_loader import get_data_loader
from app.services.ktc_utils import select_anchor_ktc, select_baseline_stats
from app.services.ktc_db import get_latest_ktc, get_latest_ktc_batch
from app.services.comps import get_comps_index
from app.schemas.player import Player, PlayerList, PlayerSummary

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api/players", tags=["players"])


@router.get("", response_model=PlayerList)
async def list_players(
    q: str = Query("", description="Search query for player name"),
    position: Optional[str] = Query(None, description="Filter by position (QB, RB, WR, TE)"),
    limit: int = Query(50, ge=1, le=2000, description="Maximum number of results"),
    sort_by: str = Query("name", regex="^(name|ktc)$", description="Sort by field"),
    sort_order: str = Query("asc", regex="^(asc|desc)$", description="Sort order"),
):
    """Search and list players."""
    data_loader = get_data_loader()
    players = data_loader.get_players()

    matching_players = []
    for player in players:
        if position and player["position"] != position:
            continue
        if q.lower() in player["name"].lower():
            matching_players.append(player)

    # Batch-fetch live KTC values from the database
    pids = [p["player_id"] for p in matching_players]
    try:
        live_ktc = await get_latest_ktc_batch(pids)
    except Exception:
        live_ktc = {}

    results = []
    for player in matching_players:
        pid = player["player_id"]
        # Prefer live KTC from DB, fall back to training data
        latest_ktc = live_ktc.get(pid)
        if latest_ktc is None:
            seasons = player.get("seasons", [])
            anchor = select_anchor_ktc(seasons) if seasons else None
            if anchor:
                latest_ktc, _, _ = anchor

        if latest_ktc is not None:
            latest_ktc = max(1.0, min(9999.0, latest_ktc))

        results.append({
            "player_id": pid,
            "name": player["name"],
            "position": player["position"],
            "latest_ktc": latest_ktc,
        })

    if sort_by == "ktc":
        results.sort(
            key=lambda x: (x["latest_ktc"] is None, -(x["latest_ktc"] or 0)),
            reverse=(sort_order == "asc"),
        )
    else:
        results.sort(
            key=lambda x: x["name"].lower(),
            reverse=(sort_order == "desc"),
        )

    results = results[:limit]

    return PlayerList(
        players=[PlayerSummary(**p) for p in results],
        total=len(results),
    )


@router.get("/positions")
def get_positions():
    return {"positions": ["QB", "RB", "WR", "TE"]}


@router.get("/{player_id}/comps")
async def player_comps(
    player_id: str,
    k: int = Query(10, ge=1, le=25, description="Number of comps"),
):
    """Find historical comparable player-seasons based on model inputs."""
    data_loader = get_data_loader()
    player = data_loader.get_player_by_id(player_id)
    if not player:
        raise HTTPException(status_code=404, detail="Player not found")

    seasons = player.get("seasons", [])
    if not seasons:
        raise HTTPException(status_code=404, detail="No season data")

    # Get current KTC (live preferred)
    try:
        live = await get_latest_ktc(player_id)
    except Exception:
        live = None

    anchor = select_anchor_ktc(seasons)
    start_ktc = live if live and live > 0 else (anchor[0] if anchor else None)
    if not start_ktc:
        raise HTTPException(status_code=404, detail="No KTC data")

    # Get latest season stats
    baseline = select_baseline_stats(seasons)
    if baseline:
        _, gp, ppg = baseline
    else:
        gp, ppg = 0, 0.0

    latest = max(seasons, key=lambda s: s["year"])
    age = latest.get("age")
    if age is None:
        raise HTTPException(status_code=404, detail="No age data")

    comps_index = get_comps_index()
    comps = comps_index.find_comps(
        position=player["position"],
        start_ktc=start_ktc,
        ppg=ppg,
        age=float(age),
        games_played=gp,
        k=k,
        exclude_player_id=player_id,
    )

    return {
        "player_id": player_id,
        "name": player["name"],
        "position": player["position"],
        "query": {
            "start_ktc": round(start_ktc, 1),
            "ppg": round(ppg, 1),
            "age": age,
            "games_played": gp,
        },
        "comps": comps,
    }


@router.get("/{player_id}", response_model=Player)
async def get_player(player_id: str):
    """Get detailed player information including all seasons."""
    data_loader = get_data_loader()
    player = data_loader.get_player_by_id(player_id)
    if not player:
        raise HTTPException(status_code=404, detail="Player not found")

    # Attach live KTC so the frontend can display the current value
    try:
        live = await get_latest_ktc(player_id)
        if live is not None:
            player = {**player, "live_ktc": live}
    except Exception:
        pass

    return Player(**player)
