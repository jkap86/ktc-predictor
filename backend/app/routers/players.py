from fastapi import APIRouter, HTTPException, Query
from typing import Optional

from app.services.data_loader import get_data_loader
from app.schemas.player import Player, PlayerList, PlayerSummary

router = APIRouter(prefix="/api/players", tags=["players"])


@router.get("", response_model=PlayerList)
def list_players(
    q: str = Query("", description="Search query for player name"),
    position: Optional[str] = Query(None, description="Filter by position (QB, RB, WR, TE)"),
    limit: int = Query(50, ge=1, le=2000, description="Maximum number of results"),
    sort_by: str = Query("name", regex="^(name|ktc)$", description="Sort by field"),
    sort_order: str = Query("asc", regex="^(asc|desc)$", description="Sort order"),
):
    """Search and list players."""
    data_loader = get_data_loader()
    results = data_loader.search_players(
        query=q,
        position=position,
        limit=limit,
        sort_by=sort_by,
        sort_order=sort_order,
    )

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
