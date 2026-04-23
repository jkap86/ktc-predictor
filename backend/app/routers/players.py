"""Player listing and detail endpoints."""

import logging
from typing import Optional

from fastapi import APIRouter, HTTPException, Query

from app.services.data_loader import get_data_loader
from app.services.ktc_utils import select_anchor_ktc, select_baseline_stats
from app.services.ktc_db import get_latest_ktc, get_latest_ktc_batch
from app.services.comps import get_comps_index
from app.services.prediction_cache import get_prediction_cache
from app.schemas.player import Player, PlayerList, PlayerSummary

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api/players", tags=["players"])


@router.get("", response_model=PlayerList)
async def list_players(
    q: str = Query("", description="Search query for player name"),
    position: Optional[str] = Query(None, description="Filter by position (QB, RB, WR, TE)"),
    limit: int = Query(50, ge=1, le=2000, description="Maximum number of results"),
    sort_by: str = Query("name", regex="^(name|ktc|predicted|change|ppg)$", description="Sort by field"),
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
        seasons = player.get("seasons", [])

        # Prefer live KTC from DB, fall back to training data
        latest_ktc = live_ktc.get(pid)
        if latest_ktc is None:
            anchor = select_anchor_ktc(seasons) if seasons else None
            if anchor:
                latest_ktc, _, _ = anchor

        if latest_ktc is not None:
            latest_ktc = max(1.0, min(9999.0, latest_ktc))
        else:
            continue  # skip players with no KTC data

        # PPG from most recent season with games
        ppg = None
        baseline = select_baseline_stats(seasons) if seasons else None
        if baseline:
            _, gp, ppg_val = baseline
            ppg = round(ppg_val, 1)

        results.append({
            "player_id": pid,
            "name": player["name"],
            "position": player["position"],
            "latest_ktc": latest_ktc,
            "ppg": ppg,
        })

    # Add cached predictions + projected PPG (precomputed at startup)
    pred_cache = get_prediction_cache()
    for r in results:
        cached = pred_cache.get(r["player_id"])
        if cached:
            r["predicted_end_ktc"] = cached["predicted_end_ktc"]
            # Use projected PPG (from Sleeper) if available, over last-season
            if cached.get("projected_ppg") is not None:
                r["ppg"] = cached["projected_ppg"]

    desc = sort_order == "desc"
    sign = -1 if desc else 1

    def _sort_key(x):
        if sort_by == "ktc":
            v = x.get("latest_ktc")
        elif sort_by == "predicted":
            v = x.get("predicted_end_ktc")
        elif sort_by == "change":
            p = x.get("predicted_end_ktc")
            k = x.get("latest_ktc")
            v = (p - k) if p is not None and k is not None else None
        elif sort_by == "ppg":
            v = x.get("ppg")
        else:
            return (0, x["name"].lower() if not desc else "")

        # Nulls always last: use inf so they sort after real values
        if v is None:
            return (1, 0)
        return (0, v * sign)

    results.sort(key=_sort_key)

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
    model: Optional[str] = Query(None, description="Model iteration ID for feature weighting"),
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
    age = latest.get("age") or 22  # default to 22 for rookies without age data

    # Pull all available features from the latest season for matching.
    # The comp finder ignores features not in its spec for the position.
    pos = player["position"]

    # Find prior season for prior-behavioral features
    sorted_seasons = sorted(seasons, key=lambda s: s["year"])
    prior = None
    for s in reversed(sorted_seasons):
        if s["year"] < latest["year"] and (s.get("games_played", 0) or 0) >= 4:
            prior = s
            break

    # KTC YoY
    ktc_yoy_pct = 0.0
    if prior and (prior.get("end_ktc") or 0) > 0:
        ktc_yoy_pct = (start_ktc - prior["end_ktc"]) / prior["end_ktc"]

    prior_ppg = 0.0
    if prior and (prior.get("games_played", 0) or 0) > 0:
        prior_ppg = (prior.get("fantasy_points", 0) or 0) / prior["games_played"]

    comps_index = get_comps_index(model)
    comps = comps_index.find_comps(
        position=pos,
        start_ktc=start_ktc,
        ppg=ppg,
        age=float(age),
        games_played=gp,
        k=k,
        exclude_player_id=player_id,
        # Core features
        weeks_missed=0,
        draft_pick=latest.get("draft_pick"),
        start_position_rank=latest.get("start_position_rank"),
        years_exp=latest.get("years_exp"),
        # KTC trajectory
        ktc_yoy_pct=ktc_yoy_pct,
        ktc_30d_trend=latest.get("ktc_30d_trend"),
        ktc_volatility=latest.get("ktc_volatility"),
        # Prior performance
        prior_ppg=prior_ppg,
        prior_year_fp=latest.get("prior_year_fp"),
        # Behavioral
        boom_rate=latest.get("boom_rate"),
        bust_rate=latest.get("bust_rate"),
        snap_pct=latest.get("snap_pct"),
        # v3 prior-behavioral
        prior_weekly_fp_cv=prior.get("weekly_fp_cv") if prior else None,
        prior_boom_rate=prior.get("boom_rate") if prior else None,
        prior_bust_rate=prior.get("bust_rate") if prior else None,
        prior_snap_pct=prior.get("snap_pct") if prior else None,
        # v4 momentum
        momentum_ratio=latest.get("momentum_ratio"),
        max_games_missed_streak=latest.get("max_games_missed_streak"),
        # Position-specific
        passing_tds=latest.get("passing_tds"),
        interceptions=latest.get("interceptions"),
        carries=latest.get("carries"),
        red_zone_touches=latest.get("red_zone_touches"),
        targets=latest.get("targets"),
        red_zone_targets=latest.get("red_zone_targets"),
    )

    return {
        "player_id": player_id,
        "name": player["name"],
        "position": pos,
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
