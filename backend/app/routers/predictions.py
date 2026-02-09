from fastapi import APIRouter, HTTPException

from app.services.model_service import get_model_service
from app.services.data_loader import get_data_loader
from app.services.transition_model_service import TransitionModelService
from app.schemas.player import (
    EOSPredictionResponse,
    EOSPredictRequest,
    CompareRequest,
    CompareResponse,
    PlayerComparison,
    TrajectoryResponse,
    WhatIfRequest,
    WhatIfResponse,
    NextWeekRequest,
    NextWeekResponse,
)

# Singleton for transition model service
_transition_service: TransitionModelService | None = None


def get_transition_service() -> TransitionModelService:
    global _transition_service
    if _transition_service is None:
        _transition_service = TransitionModelService()
    return _transition_service

router = APIRouter(prefix="/api", tags=["predictions"])


@router.get("/players/{player_id}/predict", response_model=EOSPredictionResponse)
def predict_player(player_id: str, blend_weekly: bool = False):
    """Get EOS KTC prediction for a specific player.

    Args:
        player_id: The player's unique identifier.
        blend_weekly: If true, blend EOS prediction with weekly transition model.
            Early season: 40% weekly, 60% EOS
            Late season: 10% weekly, 90% EOS
    """
    model_service = get_model_service()
    result = model_service.predict_for_player(player_id, blend_weekly=blend_weekly)

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
    """Compare multiple players with their EOS predictions.

    Args:
        request.player_ids: List of player IDs to compare.
        request.blend_weekly: If true, blend EOS prediction with weekly transition model.
    """
    if len(request.player_ids) < 2:
        raise HTTPException(status_code=400, detail="At least 2 players required")
    if len(request.player_ids) > 10:
        raise HTTPException(status_code=400, detail="Maximum 10 players allowed")

    model_service = get_model_service()
    data_loader = get_data_loader()

    players = []
    for player_id in request.player_ids:
        prediction = model_service.predict_for_player(
            player_id, blend_weekly=request.blend_weekly
        )
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


# =============================================================================
# Transition Model Endpoints (Week-by-Week Trajectory)
# =============================================================================


@router.get("/players/{player_id}/trajectory", response_model=TrajectoryResponse)
def get_trajectory(player_id: str):
    """Get week-by-week KTC trajectory for a player.

    Uses the transition model to roll out predictions from start of season.
    Returns both predicted trajectory and actual weekly KTC for comparison.
    """
    try:
        transition_service = get_transition_service()
        data_loader = get_data_loader()
        result = transition_service.predict_trajectory(player_id, data_loader)

        if not result:
            raise HTTPException(status_code=404, detail="Player not found or no trajectory data")

        return TrajectoryResponse(**result)
    except FileNotFoundError as e:
        raise HTTPException(status_code=503, detail=str(e))


@router.post("/players/{player_id}/what-if", response_model=WhatIfResponse)
def what_if_scenario(player_id: str, request: WhatIfRequest):
    """Simulate KTC trajectory with custom PPG and games scenario.

    Useful for what-if analysis: "What if this player averages 20 PPG for 15 games?"
    """
    try:
        transition_service = get_transition_service()
        data_loader = get_data_loader()
        result = transition_service.predict_what_if(
            player_id=player_id,
            data_loader=data_loader,
            target_ppg=request.ppg,
            target_games=request.games,
            current_week=request.current_week,
        )

        if not result:
            raise HTTPException(status_code=404, detail="Player not found")

        return WhatIfResponse(**result)
    except FileNotFoundError as e:
        raise HTTPException(status_code=503, detail=str(e))


@router.post("/predict/next-week", response_model=NextWeekResponse)
def predict_next_week(request: NextWeekRequest):
    """Predict next week's KTC from current state.

    Low-level single-step prediction for custom scenarios.
    """
    try:
        transition_service = get_transition_service()
        result = transition_service.predict_next_week(
            position=request.position,
            ktc_current=request.ktc_current,
            ppg_cumulative=request.ppg_cumulative,
            games_played=request.games_played,
            week=request.week,
            weekly_fp=request.weekly_fp,
            games_this_week=request.games_this_week,
            age=request.age,
        )

        return NextWeekResponse(**result)
    except FileNotFoundError as e:
        raise HTTPException(status_code=503, detail=str(e))
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
