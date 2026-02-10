from fastapi import APIRouter, HTTPException, Query
from typing import Optional

from app.services.data_loader import get_data_loader
from app.services.db import get_latest_ktc_batch
from app.services.ktc_utils import select_anchor_ktc
from app.schemas.player import Player, PlayerList, PlayerSummary

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

    # Filter by position and query
    matching_players = []
    for player in players:
        if position and player["position"] != position:
            continue
        if q.lower() in player["name"].lower():
            matching_players.append(player)

    # Fetch live KTC directly (no sync wrapper needed in async route)
    player_ids = [p["player_id"] for p in matching_players]
    live_ktc_map = await get_latest_ktc_batch(player_ids)

    # Build results with live KTC (fall back to training data if not in DB)
    results = []
    for player in matching_players:
        player_id = player["player_id"]
        live_data = live_ktc_map.get(player_id)

        if live_data and live_data.get("ktc"):
            latest_ktc = max(1.0, min(9999.0, live_data["ktc"]))
        else:
            seasons = player.get("seasons", [])
            anchor = select_anchor_ktc(seasons) if seasons else None
            if anchor:
                latest_ktc, _, _ = anchor
                if latest_ktc is not None:
                    latest_ktc = max(1.0, min(9999.0, latest_ktc))
            else:
                latest_ktc = None

        results.append({
            "player_id": player_id,
            "name": player["name"],
            "position": player["position"],
            "latest_ktc": latest_ktc,
        })

    # Sort results
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
    """Get list of available positions."""
    return {"positions": ["QB", "RB", "WR", "TE"]}


@router.get("/{player_id}", response_model=Player)
def get_player(player_id: str):
    """Get detailed player information including all seasons."""
    data_loader = get_data_loader()
    player = data_loader.get_player_by_id(player_id)

    if not player:
        raise HTTPException(status_code=404, detail="Player not found")

    return Player(**player)
