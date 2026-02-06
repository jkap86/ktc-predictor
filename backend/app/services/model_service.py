from app.config import EOS_MODELS_DIR
from app.services.data_loader import get_data_loader
from ktc_model.io import load_bundle
from ktc_model.predict import predict_end_ktc


class ModelService:
    """Lean EOS-only model service."""

    def __init__(self):
        self._bundle = None
        self._initialized = False

    def initialize(self) -> dict:
        """Load EOS model bundle from disk. Returns metrics dict."""
        self._bundle = load_bundle(str(EOS_MODELS_DIR))
        self._initialized = True
        return self._bundle.get("metrics", {})

    def _ensure_initialized(self):
        if not self._initialized:
            self.initialize()

    def predict_for_player(self, player_id: str) -> dict | None:
        """Predict end-of-season KTC for a player by ID."""
        self._ensure_initialized()

        data_loader = get_data_loader()
        player = data_loader.get_player_by_id(player_id)
        if not player:
            return None

        seasons = player.get("seasons", [])
        if not seasons:
            return None

        latest = max(seasons, key=lambda s: s["year"])
        games = latest.get("games_played", 0)
        fp = latest.get("fantasy_points", 0)
        ppg = fp / games if games > 0 else 0.0
        start_ktc = latest.get("end_ktc", latest.get("start_ktc", 0))
        age = latest.get("age")

        if start_ktc <= 0:
            return None

        result = predict_end_ktc(
            models=self._bundle["models"],
            clip_bounds=self._bundle["clip_bounds"],
            calibrators=self._bundle["calibrators"],
            position=player["position"],
            gp=games,
            ppg=ppg,
            start_ktc=start_ktc,
            age=float(age) if age is not None else None,
            sentinel_impute=self._bundle.get("sentinel_impute"),
        )

        pct_change = (result["delta_ktc"] / start_ktc * 100) if start_ktc else 0.0

        return {
            "player_id": player_id,
            "name": player["name"],
            "position": player["position"],
            "start_ktc": round(start_ktc, 1),
            "predicted_end_ktc": result["end_ktc"],
            "predicted_delta_ktc": result["delta_ktc"],
            "predicted_pct_change": round(pct_change, 2),
            "model_version": "eos_hgb_v1",
        }

    def predict_from_inputs(
        self,
        position: str,
        start_ktc: float,
        games_played: int,
        ppg: float,
        age: float | None = None,
        weeks_missed: float | None = None,
        draft_pick: float | None = None,
        years_remaining: float | None = None,
    ) -> dict:
        """Predict end-of-season KTC from raw inputs."""
        self._ensure_initialized()

        result = predict_end_ktc(
            models=self._bundle["models"],
            clip_bounds=self._bundle["clip_bounds"],
            calibrators=self._bundle["calibrators"],
            position=position,
            gp=games_played,
            ppg=ppg,
            start_ktc=start_ktc,
            age=age,
            weeks_missed=weeks_missed,
            draft_pick=draft_pick,
            years_remaining=years_remaining,
            sentinel_impute=self._bundle.get("sentinel_impute"),
        )

        pct_change = (result["delta_ktc"] / start_ktc * 100) if start_ktc else 0.0

        return {
            "position": position,
            "start_ktc": round(start_ktc, 1),
            "predicted_end_ktc": result["end_ktc"],
            "predicted_delta_ktc": result["delta_ktc"],
            "predicted_pct_change": round(pct_change, 2),
            "model_version": "eos_hgb_v1",
        }


# Singleton
_model_service = None


def get_model_service() -> ModelService:
    global _model_service
    if _model_service is None:
        _model_service = ModelService()
    return _model_service
