from typing import Optional

from fastapi import APIRouter, HTTPException

from app.services.model_service import get_model_service
from app.services.data_loader import get_data_loader
from app.schemas.player import (
    PredictionResponse,
    CompareRequest,
    CompareResponse,
    PlayerComparison,
    PredictionWithPPG,
    SimulateCurveRequest,
    SimulateCurveResponse,
    SimulationResponse,
)

router = APIRouter(prefix="/api", tags=["predictions"])


@router.get("/players/{player_id}/predict", response_model=PredictionResponse)
def predict_player(player_id: str):
    """Get KTC prediction for a specific player."""
    model_service = get_model_service()
    result = model_service.predict_for_player(player_id)

    if not result:
        raise HTTPException(status_code=404, detail="Player not found")

    return PredictionResponse(**result)


@router.post("/compare", response_model=CompareResponse)
def compare_players(request: CompareRequest):
    """Compare multiple players with their predictions."""
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
                current_ktc=prediction["current_ktc"],
                predicted_ktc=prediction["predicted_ktc"],
                ktc_change=prediction["ktc_change"],
                seasons=player_data.get("seasons", []),
            )
        )

    return CompareResponse(players=players)


@router.get("/model/metrics")
def get_model_metrics():
    """Get model performance metrics."""
    model_service = get_model_service()
    return model_service.get_metrics()


@router.get("/model/metrics-by-year")
def get_model_metrics_by_year():
    """Get model performance metrics broken down by prediction year."""
    model_service = get_model_service()
    return model_service.get_metrics_by_year()


@router.get("/model/importance")
def get_feature_importance():
    """Get feature importance from the model."""
    model_service = get_model_service()
    return model_service.get_feature_importance()


@router.get("/model/error-analysis")
def get_error_analysis():
    """Analyze prediction errors by segment (position, age bracket, KTC level).

    Returns MAE and bias (mean error) for each segment:
    - by_position: QB, RB, WR, TE
    - by_age_bracket: young, prime, declining
    - by_ktc_level: low (<2000), mid (2000-5000), high (>5000)

    Positive bias means the model is over-predicting.
    """
    model_service = get_model_service()
    return model_service.get_error_analysis()


@router.post("/model/train")
def train_model():
    """Retrain the model with current data."""
    model_service = get_model_service()
    metrics = model_service.train_model()
    return {"status": "trained", "metrics": metrics}


@router.post("/model/train-with-priors")
def train_model_with_prior_predictions():
    """Retrain the model using two-stage approach with prior predictions.

    Stage 1: Train baseline model with current features
    Stage 2: Use baseline to generate prior_predicted_ktc for training pairs
    Stage 3: Retrain model with enhanced features including prior_predicted_ktc
    """
    model_service = get_model_service()
    metrics = model_service.train_model_with_prior_predictions()
    return {"status": "trained_with_priors", "metrics": metrics}


@router.post("/model/train-ensemble")
def train_position_ensemble():
    """Train position-specific ensemble models.

    Trains separate GradientBoosting models for QB, RB, WR, and TE
    with position-specific hyperparameters for better accuracy.

    Returns metrics for each position and combined weighted average.
    """
    model_service = get_model_service()
    metrics = model_service.train_position_ensemble()
    return {"status": "trained_ensemble", "metrics": metrics}


@router.get("/model/ensemble-metrics")
def get_ensemble_metrics():
    """Get ensemble model performance metrics by position."""
    model_service = get_model_service()
    model_service.initialize_ensemble()
    return model_service.get_ensemble_metrics()


@router.post("/model/train-xgboost")
def train_xgboost_model():
    """Train XGBoost model as alternative to GradientBoosting.

    XGBoost offers better regularization (L1/L2) which may help
    reduce overfitting on high-value players.

    Requires xgboost package: pip install xgboost>=2.0.0
    """
    model_service = get_model_service()
    metrics = model_service.train_xgboost()
    if "error" in metrics:
        raise HTTPException(status_code=500, detail=metrics["error"])
    return {"status": "trained_xgboost", "metrics": metrics}


@router.get("/model/xgboost-metrics")
def get_xgboost_metrics():
    """Get XGBoost model performance metrics."""
    model_service = get_model_service()
    model_service.initialize_xgboost()
    return model_service.get_xgb_metrics()


@router.get("/model/xgboost-importance")
def get_xgboost_feature_importance():
    """Get feature importance from XGBoost model."""
    model_service = get_model_service()
    model_service.initialize_xgboost()
    return model_service.get_xgb_feature_importance()


@router.get("/model/compare")
def compare_models():
    """Compare GradientBoosting vs XGBoost performance.

    Returns metrics for both models and indicates which performs better.
    """
    model_service = get_model_service()
    return model_service.compare_models()


@router.get("/predictions/all", response_model=list[PredictionWithPPG])
def get_all_predictions(position: Optional[str] = None):
    """Get predictions for all players with PPG data."""
    model_service = get_model_service()
    data_loader = get_data_loader()

    players = data_loader.get_players()
    results = []

    for player in players:
        if position and player["position"] != position:
            continue

        prediction = model_service.predict_for_player(player["player_id"])
        if not prediction:
            continue

        # Get latest season for PPG calculation
        seasons = player.get("seasons", [])
        if not seasons:
            continue

        latest = max(seasons, key=lambda s: s["year"])
        games = latest.get("games_played", 0)
        if games == 0:
            continue

        ppg = latest.get("fantasy_points", 0) / games

        results.append(
            PredictionWithPPG(
                player_id=player["player_id"],
                name=player["name"],
                position=player["position"],
                ppg=round(ppg, 2),
                predicted_ktc=prediction["predicted_ktc"],
                current_ktc=prediction["current_ktc"],
                ktc_change_pct=prediction["ktc_change_pct"],
            )
        )

    return results


# ========== Weekly Model Endpoints ==========


@router.post("/model/train-weekly")
def train_weekly_model():
    """Train the weekly KTC change model."""
    model_service = get_model_service()
    metrics = model_service.train_weekly_model()
    return {"status": "trained_weekly", "metrics": metrics}


@router.get("/model/weekly-metrics")
def get_weekly_model_metrics():
    """Get weekly model performance metrics."""
    model_service = get_model_service()
    model_service.initialize_weekly()
    return model_service.get_weekly_metrics()


@router.get("/model/weekly-importance")
def get_weekly_feature_importance():
    """Get feature importance from the weekly model."""
    model_service = get_model_service()
    model_service.initialize_weekly()
    return model_service.get_weekly_feature_importance()


@router.post(
    "/players/{player_id}/simulate-curve",
    response_model=SimulateCurveResponse,
)
def simulate_ktc_curve(player_id: str, request: SimulateCurveRequest):
    """Generate PPG-to-KTC curve for a player over N games.

    Returns predicted KTC values for PPG 0-40, allowing the frontend
    to render a curve showing how different performance levels affect value.
    """
    if request.games < 0 or request.games > 17:
        raise HTTPException(
            status_code=400, detail="Games must be between 0 and 17"
        )

    model_service = get_model_service()
    result = model_service.simulate_curve(player_id, request.games)

    if not result:
        raise HTTPException(status_code=404, detail="Player not found")

    return SimulateCurveResponse(**result)


@router.post(
    "/players/{player_id}/simulate",
    response_model=SimulationResponse,
)
def simulate_ktc_trajectory(player_id: str, games: int, ppg: float):
    """Simulate KTC trajectory for a player over N games at given PPG.

    Returns week-by-week KTC projections.
    """
    if games < 0 or games > 17:
        raise HTTPException(
            status_code=400, detail="Games must be between 0 and 17"
        )
    if ppg < 0 or ppg > 50:
        raise HTTPException(
            status_code=400, detail="PPG must be between 0 and 50"
        )

    model_service = get_model_service()
    result = model_service.simulate_trajectory(player_id, games, ppg)

    if not result:
        raise HTTPException(status_code=404, detail="Player not found")

    return SimulationResponse(**result)
