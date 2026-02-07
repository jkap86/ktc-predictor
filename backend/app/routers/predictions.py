from fastapi import APIRouter, HTTPException

from app.services.model_service import get_model_service
from app.services.data_loader import get_data_loader
from app.schemas.player import (
    EOSPredictionResponse,
    EOSPredictRequest,
    CompareRequest,
    CompareResponse,
    PlayerComparison,
)

router = APIRouter(prefix="/api", tags=["predictions"])


@router.get("/players/{player_id}/predict", response_model=EOSPredictionResponse)
def predict_player(player_id: str):
    """Get EOS KTC prediction for a specific player."""
    model_service = get_model_service()
    result = model_service.predict_for_player(player_id)

    if not result:
        raise HTTPException(status_code=404, detail="Player not found")

    return EOSPredictionResponse(**result)


@router.post("/predict/eos", response_model=EOSPredictionResponse)
def predict_eos(request: EOSPredictRequest):
    """Predict end-of-season KTC from raw inputs."""
    model_service = get_model_service()
    try:
        result = model_service.predict_from_inputs(
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
        raise HTTPException(status_code=400, detail=str(e))

    return EOSPredictionResponse(**result)


@router.post("/compare", response_model=CompareResponse)
def compare_players(request: CompareRequest):
    """Compare multiple players with their EOS predictions."""
    if len(request.player_ids) < 2:
        raise HTTPException(status_code=400, detail="At least 2 players required")
    if len(request.player_ids) > 10:
        raise HTTPException(status_code=400, detail="Maximum 10 players allowed")

    model_service = get_model_service()
    data_loader = get_data_loader()

    players = []
    for player_id in request.player_ids:
        prediction = model_service.predict_for_player(player_id)
        if not prediction:
            continue

        player_data = data_loader.get_player_by_id(player_id)
        if not player_data:
            continue

        players.append(
            PlayerComparison(
                player_id=player_id,
                name=prediction["name"],
                position=prediction["position"],
                start_ktc=prediction["start_ktc"],
                predicted_end_ktc=prediction["predicted_end_ktc"],
                predicted_delta_ktc=prediction["predicted_delta_ktc"],
                predicted_pct_change=prediction["predicted_pct_change"],
                low_end_ktc=prediction.get("low_end_ktc"),
                high_end_ktc=prediction.get("high_end_ktc"),
                model_version=prediction["model_version"],
                anchor_year=prediction.get("anchor_year"),
                anchor_source=prediction.get("anchor_source"),
                baseline_year=prediction.get("baseline_year"),
                seasons=player_data.get("seasons", []),
            )
        )

    return CompareResponse(players=players)
