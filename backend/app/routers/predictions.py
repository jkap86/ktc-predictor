"""Prediction endpoints with model selection support."""

import logging
from typing import Optional

from fastapi import APIRouter, HTTPException, Query

from app.services.model_registry import get_registry
from app.services.data_loader import get_data_loader
from app.services.eos_model_service import predict_from_inputs, predict_for_player, predict_historical
from app.schemas.player import EOSPredictionResponse, EOSPredictRequest, HistoricalResponse, HistoricalPrediction

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
