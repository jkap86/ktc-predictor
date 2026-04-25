"""Model iteration listing endpoint."""

import json
from typing import Optional

from fastapi import APIRouter, HTTPException, Query

from app.services.model_registry import get_registry
from app.schemas.player import ModelInfo, ModelsListResponse

router = APIRouter(prefix="/api/models", tags=["models"])


@router.get("", response_model=ModelsListResponse)
def list_models():
    """List all available model iterations with their metadata and metrics."""
    registry = get_registry()
    iterations = registry.list_iterations()
    return ModelsListResponse(
        models=[ModelInfo(**it) for it in iterations],
        default_model=registry.default_id,
    )


@router.get("/diagnostics")
def model_diagnostics(
    model: Optional[str] = Query(None, description="Model iteration ID"),
):
    """Get model diagnostics: per-position MAE, bias, tier breakdowns, and direction accuracy."""
    registry = get_registry()
    try:
        iteration = registry.get(model)
    except KeyError as e:
        raise HTTPException(status_code=400, detail=str(e))

    metrics = iteration.get_metrics()

    # Load diagnostics from separate JSON file
    diag_path = iteration.models_dir / "diagnostics.json"
    diagnostics = {}
    if diag_path.exists():
        with open(diag_path) as f:
            diagnostics = json.load(f)

    return {
        "model_id": iteration.id,
        "metrics": metrics,
        "diagnostics": diagnostics,
    }
