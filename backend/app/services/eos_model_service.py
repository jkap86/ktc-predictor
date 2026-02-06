"""Core EOS model service: load artifacts, predict with confidence bands, LRU cache."""

import json
from functools import lru_cache
from pathlib import Path

import numpy as np

from app.config import EOS_MODELS_DIR, EOS_MODEL_VERSION
from ktc_model.io import load_bundle
from ktc_model.predict import predict_end_ktc


class EosModelService:
    """End-of-season KTC prediction service.

    Loads per-position XGBoost models, clip bounds, calibrators,
    sentinel imputation, and residual bands from ``EOS_MODELS_DIR``.
    """

    def __init__(self):
        self._bundle: dict | None = None
        self._residual_bands: dict = {}
        self._initialized = False

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    def initialize(self) -> dict:
        """Load model bundle. Raises FileNotFoundError if artifacts are missing."""
        models_dir = Path(EOS_MODELS_DIR)
        if not models_dir.exists():
            raise FileNotFoundError(
                f"EOS model directory not found: {models_dir}. "
                "Ensure backend/models/ contains QB.joblib, RB.joblib, WR.joblib, TE.joblib, "
                "clip_bounds.json, and calibrators/."
            )

        required = ["QB.joblib", "RB.joblib", "WR.joblib", "TE.joblib", "clip_bounds.json"]
        missing = [f for f in required if not (models_dir / f).exists()]
        if missing:
            raise FileNotFoundError(
                f"Missing required EOS artifacts in {models_dir}: {missing}"
            )

        self._bundle = load_bundle(str(models_dir))
        self._residual_bands = self._bundle.get("residual_bands", {})
        self._initialized = True

        # Clear LRU cache on re-init
        self._cached_predict.cache_clear()

        return self._bundle.get("metrics", {})

    @property
    def bundle(self) -> dict:
        if not self._initialized:
            self.initialize()
        assert self._bundle is not None
        return self._bundle

    # ------------------------------------------------------------------
    # Prediction (with LRU cache)
    # ------------------------------------------------------------------

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
        """Predict EOS KTC from raw inputs. Cached by input tuple."""
        return self._cached_predict(
            position, start_ktc, games_played, ppg,
            age, weeks_missed, draft_pick, years_remaining,
        )

    @lru_cache(maxsize=2048)
    def _cached_predict(
        self,
        position: str,
        start_ktc: float,
        games_played: int,
        ppg: float,
        age: float | None,
        weeks_missed: float | None,
        draft_pick: float | None,
        years_remaining: float | None,
    ) -> dict:
        b = self.bundle
        result = predict_end_ktc(
            models=b["models"],
            clip_bounds=b["clip_bounds"],
            calibrators=b["calibrators"],
            position=position,
            gp=games_played,
            ppg=ppg,
            start_ktc=start_ktc,
            age=age,
            weeks_missed=weeks_missed,
            draft_pick=draft_pick,
            years_remaining=years_remaining,
            sentinel_impute=b.get("sentinel_impute"),
        )

        pct = (result["delta_ktc"] / start_ktc * 100) if start_ktc else 0.0

        # Confidence bands from residual percentiles
        bands = self._residual_bands.get(position, {})
        low_end_ktc = None
        high_end_ktc = None
        if bands and start_ktc > 0:
            pred_log = np.log(result["end_ktc"] / start_ktc)
            low_end_ktc = round(start_ktc * np.exp(pred_log + bands["p20"]), 1)
            high_end_ktc = round(start_ktc * np.exp(pred_log + bands["p80"]), 1)

        return {
            "position": position,
            "start_ktc": round(start_ktc, 1),
            "predicted_end_ktc": result["end_ktc"],
            "predicted_delta_ktc": result["delta_ktc"],
            "predicted_pct_change": round(pct, 2),
            "low_end_ktc": low_end_ktc,
            "high_end_ktc": high_end_ktc,
            "model_version": EOS_MODEL_VERSION,
        }

    # ------------------------------------------------------------------
    # Player-level prediction
    # ------------------------------------------------------------------

    def predict_for_player(self, player_id: str, data_loader) -> dict | None:
        """Predict EOS KTC for a player using their latest season data."""
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

        result = self.predict_from_inputs(
            position=player["position"],
            start_ktc=start_ktc,
            games_played=games,
            ppg=ppg,
            age=float(age) if age is not None else None,
        )
        result["player_id"] = player_id
        result["name"] = player["name"]
        return result
