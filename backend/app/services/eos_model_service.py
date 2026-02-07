"""Core EOS model service: load artifacts, predict with confidence bands, LRU cache."""

import json
from functools import lru_cache
from pathlib import Path

import numpy as np

from app.config import EOS_MODELS_DIR, EOS_MODEL_VERSION
from app.services.ktc_utils import _is_valid_ktc, select_anchor_ktc, select_baseline_stats
from ktc_model.io import load_bundle
from ktc_model.predict import predict_end_ktc


def _cap_ktc(x):
    """Clamp KTC value to valid domain [1, 9999]."""
    if x is None:
        return None
    return max(1.0, min(9999.0, x))


class EosModelService:
    """End-of-season KTC prediction service.

    Loads per-position gradient boosting models (HGB or XGBoost), clip bounds,
    linear calibrators, sentinel imputation, and residual bands from ``EOS_MODELS_DIR``.
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
        """Predict EOS KTC from raw inputs. Cached by quantized input tuple."""
        # Quantize floats for cache efficiency (reduces key explosion from slider UX)
        q_start_ktc = round(start_ktc / 5) * 5  # nearest 5
        q_ppg = round(ppg, 1)  # 1 decimal
        q_age = round(age, 1) if age is not None else None

        return self._cached_predict(
            position, q_start_ktc, games_played, q_ppg,
            q_age, weeks_missed, draft_pick, years_remaining,
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

        # Use effective (post-imputation) start_ktc for display + math
        effective_ktc = result.get("effective_start_ktc", start_ktc)

        # Confidence bands from residual percentiles
        bands = self._residual_bands.get(position, {})
        low_end_ktc = None
        high_end_ktc = None
        if bands and effective_ktc > 0:
            pred_log = np.log(result["end_ktc"] / effective_ktc)
            low_end_ktc = round(effective_ktc * np.exp(pred_log + bands["p20"]), 1)
            high_end_ktc = round(effective_ktc * np.exp(pred_log + bands["p80"]), 1)

        # Clamp all KTC outputs to valid domain [1, 9999]
        predicted_end_ktc = _cap_ktc(result["end_ktc"])
        low_end_ktc = _cap_ktc(low_end_ktc)
        high_end_ktc = _cap_ktc(high_end_ktc)

        # Recompute delta and pct using clamped predicted_end_ktc
        delta_ktc = predicted_end_ktc - effective_ktc if predicted_end_ktc else result["delta_ktc"]
        pct = (delta_ktc / effective_ktc * 100) if effective_ktc else 0.0

        return {
            "position": position,
            "start_ktc": round(effective_ktc, 1),
            "predicted_end_ktc": predicted_end_ktc,
            "predicted_delta_ktc": round(delta_ktc, 1),
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

        # --- anchor KTC (shared logic) ---
        anchor = select_anchor_ktc(seasons)
        if anchor is None:
            return None
        start_ktc, anchor_year, anchor_source = anchor

        # --- baseline stats (shared logic) ---
        latest = max(seasons, key=lambda s: s["year"])
        baseline_info = select_baseline_stats(seasons)
        if baseline_info:
            baseline_year, games, ppg = baseline_info
        else:
            baseline_year = latest["year"]
            games = 0
            ppg = 0.0

        # Age: prefer baseline season, fall back to latest
        baseline_season = next(
            (s for s in seasons if s["year"] == baseline_year), latest
        )
        age = baseline_season.get("age") or latest.get("age")

        result = self.predict_from_inputs(
            position=player["position"],
            start_ktc=start_ktc,
            games_played=games,
            ppg=ppg,
            age=float(age) if age is not None else None,
        )
        result["player_id"] = player_id
        result["name"] = player["name"]
        result["anchor_year"] = anchor_year
        result["anchor_source"] = anchor_source
        result["baseline_year"] = baseline_year
        return result
