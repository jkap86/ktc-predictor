"""Thin singleton wrapper around EosModelService."""

from app.services.eos_model_service import EosModelService
from app.services.data_loader import get_data_loader


class ModelService:
    """Application-level model service (singleton facade)."""

    def __init__(self):
        self._eos = EosModelService()

    def initialize(self) -> dict:
        return self._eos.initialize()

    def predict_for_player(self, player_id: str) -> dict | None:
        return self._eos.predict_for_player(player_id, get_data_loader())

    def predict_from_inputs(self, **kwargs) -> dict:
        return self._eos.predict_from_inputs(**kwargs)


_model_service = None


def get_model_service() -> ModelService:
    global _model_service
    if _model_service is None:
        _model_service = ModelService()
    return _model_service
