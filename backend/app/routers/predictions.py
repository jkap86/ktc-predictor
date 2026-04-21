"""Prediction endpoints with model selection support."""

import logging
from typing import Optional

from fastapi import APIRouter, HTTPException, Query

from app.services.model_registry import get_registry
from app.services.data_loader import get_data_loader
from app.services.eos_model_service import predict_from_inputs, predict_for_player, predict_for_player_whatif, predict_historical
from app.services.ktc_db import get_latest_ktc_batch
from app.schemas.player import EOSPredictionResponse, EOSPredictRequest, EOSBatchRequest, EOSBatchResponse, PlayerWhatIfBatchRequest, HistoricalResponse, HistoricalPrediction

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api", tags=["predictions"])


@router.get("/players/{player_id}/predict", response_model=EOSPredictionResponse)
async def predict_player(
    player_id: str,
    model: Optional[str] = Query(None, description="Model iteration ID"),
):
    """Get EOS KTC prediction for a specific player."""
    registry = get_registry()
    try:
        iteration = registry.get(model)
    except KeyError as e:
        raise HTTPException(status_code=400, detail=str(e))

    data_loader = get_data_loader()
    result = await predict_for_player(iteration, player_id, data_loader)

    if not result:
        raise HTTPException(status_code=404, detail="Player not found")

    return EOSPredictionResponse(**result)


@router.post("/players/{player_id}/predict/whatif-batch", response_model=EOSBatchResponse)
async def predict_player_whatif_batch(
    player_id: str,
    request: PlayerWhatIfBatchRequest,
    model: Optional[str] = Query(None, description="Model iteration ID"),
):
    """Predict EOS KTC for a player with overridden games/ppg.

    Uses all the player's real context (prior stats, momentum, team, etc.)
    but lets the caller sweep over PPG values for What-If charts.
    """
    registry = get_registry()
    try:
        iteration = registry.get(model)
    except KeyError as e:
        raise HTTPException(status_code=400, detail=str(e))

    data_loader = get_data_loader()
    predictions = []
    for ppg in request.ppg_values:
        try:
            result = await predict_for_player_whatif(
                iteration, player_id, data_loader,
                games_played=request.games_played, ppg=ppg,
            )
            if result:
                predictions.append(EOSPredictionResponse(**result))
            else:
                raise ValueError("Player not found")
        except Exception:
            predictions.append(EOSPredictionResponse(
                position="QB", start_ktc=0,
                predicted_end_ktc=0, predicted_delta_ktc=0, predicted_pct_change=0,
            ))

    return EOSBatchResponse(predictions=predictions, model_version=iteration.id)


@router.get("/players/{player_id}/historical", response_model=HistoricalResponse)
def player_historical(player_id: str):
    """Get historical predictions vs actuals across all temporal model variants."""
    data_loader = get_data_loader()
    player = data_loader.get_player_by_id(player_id)
    if not player:
        raise HTTPException(status_code=404, detail="Player not found")

    predictions = predict_historical(player_id, data_loader)
    return HistoricalResponse(
        player_id=player_id,
        name=player["name"],
        position=player["position"],
        predictions=[HistoricalPrediction(**p) for p in predictions],
    )


@router.post("/predict/eos/batch", response_model=EOSBatchResponse)
def predict_eos_batch(
    request: EOSBatchRequest,
    model: Optional[str] = Query(None, description="Model iteration ID"),
):
    """Predict EOS KTC for multiple PPG values in one call (used by What-If chart)."""
    registry = get_registry()
    try:
        iteration = registry.get(model)
    except KeyError as e:
        raise HTTPException(status_code=400, detail=str(e))

    predictions = []
    for ppg in request.ppg_values:
        try:
            result = predict_from_inputs(
                iteration=iteration,
                position=request.position,
                start_ktc=request.start_ktc,
                games_played=request.games_played,
                ppg=ppg,
                age=request.age,
                weeks_missed=request.weeks_missed,
                draft_pick=request.draft_pick,
                years_remaining=request.years_remaining,
            )
            predictions.append(EOSPredictionResponse(**result))
        except Exception:
            predictions.append(EOSPredictionResponse(
                position=request.position,
                start_ktc=request.start_ktc,
                predicted_end_ktc=0,
                predicted_delta_ktc=0,
                predicted_pct_change=0,
            ))

    return EOSBatchResponse(
        predictions=predictions,
        model_version=iteration.id,
    )


@router.post("/predict/eos", response_model=EOSPredictionResponse)
def predict_eos(
    request: EOSPredictRequest,
    model: Optional[str] = Query(None, description="Model iteration ID"),
):
    """Predict end-of-season KTC from raw inputs."""
    registry = get_registry()
    try:
        iteration = registry.get(model)
    except KeyError as e:
        raise HTTPException(status_code=400, detail=str(e))

    try:
        result = predict_from_inputs(
            iteration=iteration,
            position=request.position,
            start_ktc=request.start_ktc,
            games_played=request.games_played,
            ppg=request.ppg,
            age=request.age,
            weeks_missed=request.weeks_missed,
            draft_pick=request.draft_pick,
            years_remaining=request.years_remaining,
        )
    except (ValueError, KeyError) as e:
        logger.warning("EOS prediction input error: %s", e)
        raise HTTPException(status_code=400, detail="Invalid prediction inputs")

    return EOSPredictionResponse(**result)


@router.get("/top-movers")
async def top_movers(
    limit: int = Query(20, ge=1, le=100),
    position: Optional[str] = Query(None),
):
    """Get players with the biggest predicted EOS changes (risers + fallers)."""
    registry = get_registry()
    iteration = registry.get()
    data_loader = get_data_loader()
    players = data_loader.get_players()

    valid_positions = {"QB", "RB", "WR", "TE"}
    candidates = [
        p for p in players
        if p["position"] in valid_positions
        and (position is None or p["position"] == position)
        and p.get("seasons")
    ]

    # Batch-fetch live KTC
    pids = [p["player_id"] for p in candidates]
    try:
        live_ktc = await get_latest_ktc_batch(pids)
    except Exception:
        live_ktc = {}

    # Only predict for players that have live KTC
    results = []
    for player in candidates:
        pid = player["player_id"]
        ktc = live_ktc.get(pid)
        if not ktc or ktc <= 0:
            continue

        try:
            pred = await predict_for_player(iteration, pid, data_loader)
            if pred:
                results.append({
                    "player_id": pid,
                    "name": player["name"],
                    "position": player["position"],
                    "start_ktc": pred["start_ktc"],
                    "predicted_end_ktc": pred["predicted_end_ktc"],
                    "predicted_delta_ktc": pred["predicted_delta_ktc"],
                    "predicted_pct_change": pred["predicted_pct_change"],
                })
        except Exception:
            continue

    results.sort(key=lambda r: r["predicted_pct_change"], reverse=True)
    risers = results[:limit]
    fallers = results[-limit:][::-1] if len(results) > limit else []

    return {"risers": risers, "fallers": fallers}
