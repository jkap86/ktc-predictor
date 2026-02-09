"""Thin singleton wrapper around EosModelService."""

from app.services.eos_model_service import EosModelService
from app.services.data_loader import get_data_loader
from app.services.transition_model_service import TransitionModelService


class ModelService:
    """Application-level model service (singleton facade)."""

    def __init__(self):
        self._eos = EosModelService()
        self._transition: TransitionModelService | None = None

    def initialize(self) -> dict:
        return self._eos.initialize()

    def _get_transition_service(self) -> TransitionModelService:
        if self._transition is None:
            self._transition = TransitionModelService()
        return self._transition

    def predict_for_player(self, player_id: str, blend_weekly: bool = False) -> dict | None:
        data_loader = get_data_loader()
        if blend_weekly:
            try:
                transition_service = self._get_transition_service()
                return self._eos.predict_with_weekly_blend(
                    player_id, data_loader, transition_service
                )
            except FileNotFoundError:
                # Fall back to EOS-only if transition models not available
                return self._eos.predict_for_player(player_id, data_loader)
        return self._eos.predict_for_player(player_id, data_loader)

    def predict_from_inputs(self, **kwargs) -> dict:
        return self._eos.predict_from_inputs(**kwargs)


_model_service = None


def get_model_service() -> ModelService:
    global _model_service
    if _model_service is None:
        _model_service = ModelService()
    return _model_service
