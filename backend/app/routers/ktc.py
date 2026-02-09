"""Live KTC endpoints with hourly caching."""
from fastapi import APIRouter, HTTPException
from pydantic import BaseModel
from datetime import date
from typing import Optional

from app.services.db import get_latest_ktc, get_latest_ktc_batch, get_cache_stats

router = APIRouter(prefix="/api/ktc", tags=["ktc"])


class LiveKTC(BaseModel):
    player_id: str
    ktc: int
    date: date
    overall_rank: Optional[int]
    position_rank: Optional[int]


@router.get("/stats")
async def cache_stats():
    """Get cache statistics."""
    return get_cache_stats()


@router.get("/{player_id}", response_model=LiveKTC)
async def get_player_ktc(player_id: str):
    """Get live KTC value for a player (cached hourly)."""
    result = await get_latest_ktc(player_id)
    if not result:
        raise HTTPException(status_code=404, detail="Player KTC not found")
    return LiveKTC(player_id=player_id, **result)


@router.post("/batch", response_model=list[LiveKTC])
async def get_batch_ktc(player_ids: list[str]):
    """Get live KTC for multiple players (cached hourly)."""
    results = await get_latest_ktc_batch(player_ids)
    return [
        LiveKTC(player_id=pid, **data)
        for pid, data in results.items()
    ]
