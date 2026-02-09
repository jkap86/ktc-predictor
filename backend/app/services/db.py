"""Database connection and KTC queries with hourly caching."""
import os
import asyncio
from datetime import datetime, timedelta
from typing import Optional
import asyncpg
from dotenv import load_dotenv

load_dotenv()

DATABASE_URL = os.getenv("DATABASE_URL")

# Connection pool
_pool: Optional[asyncpg.Pool] = None

# Cache: player_id -> {ktc, date, overall_rank, position_rank, cached_at}
_ktc_cache: dict[str, dict] = {}
_cache_lock = asyncio.Lock()
CACHE_TTL = timedelta(hours=1)


async def get_pool() -> asyncpg.Pool:
    """Get or create connection pool."""
    global _pool
    if _pool is None:
        _pool = await asyncpg.create_pool(DATABASE_URL, min_size=2, max_size=10)
    return _pool


def _is_cache_valid(cached_at: datetime) -> bool:
    """Check if cache entry is still valid (< 1 hour old)."""
    return datetime.now() - cached_at < CACHE_TTL


async def get_latest_ktc(player_id: str) -> dict | None:
    """Get most recent KTC value for a player (cached hourly)."""
    # Check cache first
    async with _cache_lock:
        if player_id in _ktc_cache:
            entry = _ktc_cache[player_id]
            if _is_cache_valid(entry["cached_at"]):
                return {k: v for k, v in entry.items() if k != "cached_at"}

    # Cache miss or expired - fetch from DB
    pool = await get_pool()
    async with pool.acquire() as conn:
        row = await conn.fetchrow(
            """
            SELECT value, date, overall_rank, position_rank
            FROM ktc_dynasty
            WHERE player_id = $1
            ORDER BY date DESC
            LIMIT 1
            """,
            player_id
        )
        if row:
            result = {
                "ktc": row["value"],
                "date": row["date"],
                "overall_rank": row["overall_rank"],
                "position_rank": row["position_rank"],
            }
            # Update cache
            async with _cache_lock:
                _ktc_cache[player_id] = {**result, "cached_at": datetime.now()}
            return result
    return None


async def get_latest_ktc_batch(player_ids: list[str]) -> dict[str, dict]:
    """Get latest KTC for multiple players (cached hourly)."""
    results = {}
    uncached_ids = []

    # Check cache for each player
    async with _cache_lock:
        for pid in player_ids:
            if pid in _ktc_cache:
                entry = _ktc_cache[pid]
                if _is_cache_valid(entry["cached_at"]):
                    results[pid] = {k: v for k, v in entry.items() if k != "cached_at"}
                else:
                    uncached_ids.append(pid)
            else:
                uncached_ids.append(pid)

    # Fetch uncached from DB
    if uncached_ids:
        pool = await get_pool()
        async with pool.acquire() as conn:
            rows = await conn.fetch(
                """
                SELECT DISTINCT ON (player_id)
                    player_id, value, date, overall_rank, position_rank
                FROM ktc_dynasty
                WHERE player_id = ANY($1)
                ORDER BY player_id, date DESC
                """,
                uncached_ids
            )
            now = datetime.now()
            async with _cache_lock:
                for row in rows:
                    entry = {
                        "ktc": row["value"],
                        "date": row["date"],
                        "overall_rank": row["overall_rank"],
                        "position_rank": row["position_rank"],
                    }
                    _ktc_cache[row["player_id"]] = {**entry, "cached_at": now}
                    results[row["player_id"]] = entry

    return results


async def invalidate_cache(player_id: Optional[str] = None):
    """Invalidate cache for a player or all players."""
    async with _cache_lock:
        if player_id:
            _ktc_cache.pop(player_id, None)
        else:
            _ktc_cache.clear()


def get_cache_stats() -> dict:
    """Get cache statistics for monitoring."""
    now = datetime.now()
    valid = sum(1 for e in _ktc_cache.values() if _is_cache_valid(e["cached_at"]))
    return {
        "total_entries": len(_ktc_cache),
        "valid_entries": valid,
        "expired_entries": len(_ktc_cache) - valid,
    }
