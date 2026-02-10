"""Core EOS model service: load artifacts, predict with confidence bands, LRU cache."""

import asyncio
import concurrent.futures
import json
import logging
from functools import lru_cache
from pathlib import Path

import numpy as np

from app.config import EOS_MODELS_DIR, EOS_MODEL_VERSION
from app.services.ktc_utils import (
    _is_valid_ktc,
    compute_prior_ktc_features,
    compute_prior_ppg,
    select_anchor_ktc,
    select_baseline_stats,
)
from ktc_model.io import load_bundle
from ktc_model.predict import predict_end_ktc, validate_feature_contract

logger = logging.getLogger(__name__)


def get_live_ktc_sync(player_id: str) -> dict | None:
    """Synchronous wrapper to fetch live KTC from database.

    Returns dict with 'ktc', 'date', 'overall_rank', 'position_rank' or None.
    """
    from app.services.db import get_latest_ktc

    try:
        # Create a new event loop in a thread for sync context
        with concurrent.futures.ThreadPoolExecutor() as pool:
            future = pool.submit(asyncio.run, get_latest_ktc(player_id))
            return future.result(timeout=5)
    except Exception as e:
        logger.warning("Failed to fetch live KTC for player %s: %s", player_id, e)
        return None


def _cap_ktc(x):
    """Clamp KTC value to valid domain [1, 9999]."""
    if x is None:
        return None
    return max(1.0, min(9999.0, x))


# Tier-specific confidence band multipliers
# Elite tiers (especially RB 6k+) have higher inherent uncertainty
# due to the difficulty of predicting whether elite players will rise or fall
_BAND_MULTIPLIERS = {
    "RB": {
        # RB 6k+ tier has 1164 riser bias - inherently unpredictable
        6000: 1.6,  # 60% wider bands for elite RBs
        4000: 1.2,  # 20% wider for high-tier RBs
    },
    "QB": {
        # QB 6k+ tier has 706 riser bias
        6000: 1.4,  # 40% wider bands for elite QBs
        4000: 1.15,  # 15% wider for high-tier QBs
    },
}


def _get_band_multiplier(position: str, start_ktc: float) -> float:
    """Get confidence band multiplier based on position and tier.

    Elite tiers have wider bands due to higher prediction uncertainty.
    """
    if position not in _BAND_MULTIPLIERS:
        return 1.0

    thresholds = _BAND_MULTIPLIERS[position]
    # Check thresholds from highest to lowest
    for threshold in sorted(thresholds.keys(), reverse=True):
        if start_ktc >= threshold:
            return thresholds[threshold]

    return 1.0


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

        # Validate feature contract to catch train/predict mismatches early
        validate_feature_contract(self._bundle.get("feature_names"))

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
        prior_end_ktc: float | None = None,
        max_ktc_prior: float | None = None,
        prior_ppg: float | None = None,
    ) -> dict:
        """Predict EOS KTC from raw inputs. Cached by quantized input tuple.

        For QB predictions, prior_end_ktc, max_ktc_prior, and prior_ppg enable
        trajectory features that improve accuracy.
        """
        # Quantize floats for cache efficiency (reduces key explosion from slider UX)
        q_start_ktc = round(start_ktc / 5) * 5  # nearest 5
        q_ppg = round(ppg, 1)  # 1 decimal
        q_age = round(age, 1) if age is not None else None
        # Quantize prior KTC/PPG values too (only used for QB)
        q_prior_end_ktc = round(prior_end_ktc / 10) * 10 if prior_end_ktc else None
        q_max_ktc_prior = round(max_ktc_prior / 10) * 10 if max_ktc_prior else None
        q_prior_ppg = round(prior_ppg, 1) if prior_ppg else None

        return self._cached_predict(
            position, q_start_ktc, games_played, q_ppg,
            q_age, weeks_missed, draft_pick, years_remaining,
            q_prior_end_ktc, q_max_ktc_prior, q_prior_ppg,
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
        prior_end_ktc: float | None = None,
        max_ktc_prior: float | None = None,
        prior_ppg: float | None = None,
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
            prior_end_ktc=prior_end_ktc,
            max_ktc_prior=max_ktc_prior,
            prior_ppg=prior_ppg,
            sentinel_impute=b.get("sentinel_impute"),
            residual_correction=b.get("residual_correction"),
            knn_adjuster=b.get("knn_adjuster"),
            target_type=b.get("target_type", "log_ratio"),
        )

        # Use effective (post-imputation) start_ktc for display + math
        effective_ktc = result.get("effective_start_ktc", start_ktc)

        # Confidence bands from residual percentiles
        # Apply tier-specific multiplier to widen bands for elite tiers
        bands = self._residual_bands.get(position, {})
        low_end_ktc = None
        high_end_ktc = None
        if bands and effective_ktc > 0:
            pred_log = np.log(result["end_ktc"] / effective_ktc)
            multiplier = _get_band_multiplier(position, effective_ktc)
            # Widen bands by multiplier (negative p20 becomes more negative, positive p80 stays positive)
            low_end_ktc = round(effective_ktc * np.exp(pred_log + bands["p20"] * multiplier), 1)
            high_end_ktc = round(effective_ktc * np.exp(pred_log + bands["p80"] * multiplier), 1)

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

        # --- anchor KTC: prefer live DB, fall back to training data ---
        live_ktc = get_live_ktc_sync(player_id)
        if live_ktc and _is_valid_ktc(live_ktc.get("ktc")):
            start_ktc = live_ktc["ktc"]
            anchor_year = None  # Live KTC doesn't have a season year
            anchor_source = "live_db"
        else:
            # Fall back to historical training data
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

        # Prior-season features for stable trajectory positions (QB, WR, TE)
        # RB excluded due to high variance at elite tier
        prior_end_ktc = None
        max_ktc_prior = None
        prior_ppg = None
        # For prior features, use anchor_year if available, else use year after latest season
        prior_ref_year = anchor_year if anchor_year else (latest["year"] + 1)
        if player["position"] in ("QB", "WR", "TE"):
            prior_end_ktc, max_ktc_prior = compute_prior_ktc_features(
                seasons, prior_ref_year
            )
            prior_ppg = compute_prior_ppg(seasons, prior_ref_year)

        result = self.predict_from_inputs(
            position=player["position"],
            start_ktc=start_ktc,
            games_played=games,
            ppg=ppg,
            age=float(age) if age is not None else None,
            prior_end_ktc=prior_end_ktc,
            max_ktc_prior=max_ktc_prior,
            prior_ppg=prior_ppg,
        )
        result["player_id"] = player_id
        result["name"] = player["name"]
        result["anchor_year"] = anchor_year
        result["anchor_source"] = anchor_source
        result["baseline_year"] = baseline_year
        return result

    # ------------------------------------------------------------------
    # Blended prediction (EOS + weekly rollout)
    # ------------------------------------------------------------------

    def predict_with_weekly_blend(
        self,
        player_id: str,
        data_loader,
        transition_service,
    ) -> dict | None:
        """Predict EOS KTC blending EOS model with weekly rollout.

        The weekly rollout model is more reactive to recent performance,
        while the EOS model provides a stable end-of-season view.
        Early-season: trust weekly more (captures hot/cold streaks)
        Late-season: trust EOS more (stable projection)

        Parameters
        ----------
        player_id : str
            Player ID to predict for.
        data_loader : DataLoaderService
            Data loader for player data.
        transition_service : TransitionModelService
            Weekly transition model service.

        Returns
        -------
        dict or None
            Blended prediction result with:
            - predicted_end_ktc: blended EOS + weekly
            - eos_end_ktc: raw EOS prediction
            - weekly_end_ktc: raw weekly rollout prediction
            - blend_weight: weight given to weekly model
        """
        # Get EOS prediction
        eos_result = self.predict_for_player(player_id, data_loader)
        if eos_result is None:
            return None

        # Get weekly rollout prediction
        try:
            weekly_result = transition_service.predict_trajectory(player_id, data_loader)
        except Exception:
            weekly_result = None

        if weekly_result is None:
            # Fall back to EOS-only if weekly model unavailable
            return eos_result

        # Get games played for blend weighting
        player = data_loader.get_player_by_id(player_id)
        seasons = player.get("seasons", [])
        latest = max(seasons, key=lambda s: s["year"]) if seasons else {}
        weekly_stats = latest.get("weekly_stats", [])
        games_played = sum(ws.get("games_played", 0) for ws in weekly_stats)

        # Blend weights: early season trusts weekly more, late season trusts EOS
        if games_played <= 4:
            weekly_weight = 0.4  # 40% weekly, 60% EOS
        elif games_played <= 8:
            weekly_weight = 0.3
        elif games_played <= 12:
            weekly_weight = 0.2
        else:
            weekly_weight = 0.1  # Late season: mostly EOS

        # Blend predictions
        eos_end = eos_result["predicted_end_ktc"]
        weekly_end = weekly_result["end_ktc"]
        blended_end = (1 - weekly_weight) * eos_end + weekly_weight * weekly_end

        # Clamp to valid domain
        blended_end = max(1.0, min(9999.0, blended_end))

        # Create result with both predictions
        result = eos_result.copy()
        result["predicted_end_ktc"] = round(blended_end, 1)
        result["predicted_delta_ktc"] = round(blended_end - result["start_ktc"], 1)
        result["predicted_pct_change"] = round(
            (blended_end - result["start_ktc"]) / result["start_ktc"] * 100, 2
        )
        result["eos_end_ktc"] = eos_end
        result["weekly_end_ktc"] = round(weekly_end, 1)
        result["blend_weight"] = weekly_weight
        result["games_played"] = games_played

        return result
