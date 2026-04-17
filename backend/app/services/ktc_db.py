"""Fetch live KTC values from the postgres ktc_dynasty table."""

import os
import logging
from typing import Optional

import asyncpg

logger = logging.getLogger(__name__)

_pool: Optional[asyncpg.Pool] = None


async def get_pool() -> asyncpg.Pool:
    global _pool
    if _pool is None:
        db_url = os.getenv("DATABASE_URL")
        if not db_url:
            raise RuntimeError("DATABASE_URL not set")
        _pool = await asyncpg.create_pool(db_url, min_size=1, max_size=5)
    return _pool


async def close_pool():
    global _pool
    if _pool:
        await _pool.close()
        _pool = None


async def get_latest_ktc(player_id: str) -> Optional[float]:
    """Return the most recent KTC value for a player, or None."""
    pool = await get_pool()
    row = await pool.fetchrow(
        "SELECT value FROM ktc_dynasty WHERE player_id = $1 ORDER BY date DESC LIMIT 1",
        player_id,
    )
    return float(row["value"]) if row else None


async def get_latest_ktc_batch(player_ids: list[str]) -> dict[str, float]:
    """Return latest KTC values for multiple players at once."""
    if not player_ids:
        return {}
    pool = await get_pool()
    rows = await pool.fetch(
        """
        SELECT DISTINCT ON (player_id) player_id, value
        FROM ktc_dynasty
        WHERE player_id = ANY($1)
        ORDER BY player_id, date DESC
        """,
        player_ids,
    )
    return {r["player_id"]: float(r["value"]) for r in rows}
