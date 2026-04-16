"""Model iteration listing endpoint."""

from fastapi import APIRouter

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
