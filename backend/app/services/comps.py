"""Find historical comparable players using the active model's feature importances.

Builds per-position KNN indices over training data, weighted by the actual
feature importances extracted from the trained model's tree split counts.
When the user switches models, the comp matching automatically adjusts to
reflect what that model considers important.
"""

import json
import logging
import zipfile
from collections import Counter
from pathlib import Path
from typing import Optional

import numpy as np
from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import StandardScaler

from app.config import TRAINING_DATA_PATH

logger = logging.getLogger(__name__)

_PRIME_AGE = {"QB": 27, "RB": 24, "WR": 26, "TE": 27}


def _get_ktc_quartile(ktc: float) -> int:
    if ktc < 1559: return 1
    elif ktc < 3085: return 2
    elif ktc < 4850: return 3
    return 4


def _age_prime_distance(age: float, position: str) -> float:
    return age - _PRIME_AGE.get(position, 26)


def _extract_feature_importances(model, n_features: int) -> np.ndarray:
    """Extract feature importance from an HGB ensemble by counting tree splits.

    Works with both EnsembleModel (list of HGBs) and raw HGBRegressors.
    Returns normalized importance weights (sum to 1).
    """
    counts = np.zeros(n_features)

    models = model.models if hasattr(model, 'models') else [model]
    for m in models:
        if not hasattr(m, '_predictors'):
            continue
        for iteration_predictors in m._predictors:
            for tree in iteration_predictors:
                for node in tree.nodes:
                    if node['is_leaf'] == 0:
                        idx = node['feature_idx']
                        if 0 <= idx < n_features:
                            counts[idx] += 1

    total = counts.sum()
    if total > 0:
        return counts / total
    # Fallback: uniform weights
    return np.ones(n_features) / n_features


# ── All features we can compute from training data ──────────────────────

# This is the superset of features any model version might use.
# We compute all of them for every player-season. When building the KNN
# index for a specific model, we select only the features that model uses
# and weight them by that model's importances.

# Pre-season only features — no current-season PPG/games/performance.
# Comps should match on who the player is going INTO the season,
# so comp avg PPG can serve as an unbiased PPG projection.
_ALL_SEASON_FEATURES = [
    "draft_pick",
    "years_remaining",
    "start_ktc_quartile",
    "age_prime_distance",
    "start_position_rank",
    "start_ktc",
    "start_ktc_was_sentinel",
    "years_exp",
    "ktc_yoy_log",
    "ktc_peak_drawdown",
    "has_prior_season",
    "prior_ppg",
    "has_prior_ppg",
    "apy_cap_pct",
    "is_contract_year",
    "apy_position_rank",
    "has_contract_data",
    # v3 behavioral (prior season)
    "prior_weekly_fp_cv",
    "prior_boom_rate",
    "prior_bust_rate",
    "prior_snap_pct",
    "prior_ktc_volatility",
    "has_prior_behavioral",
    # v4 momentum (pre-season KTC trends)
    "ktc_30d_trend",
    "ktc_90d_trend",
    "momentum_ratio",
    "max_games_missed_streak",
    # Position-specific (prior season)
    "prior_passing_tds",
    "prior_interceptions",
    "prior_carries",
    "prior_red_zone_touches",
    "prior_targets",
    "prior_red_zone_targets",
    "has_prior_position_stats",
]


def _compute_all_features(season: dict, prior: dict | None, position: str, start_ktc: float, ppg: float, gp: int, age: float) -> dict[str, float]:
    """Compute pre-season features from a season record.

    Only uses data known before the season starts (KTC, age, prior stats).
    Current-season PPG/games are excluded so comps match on profile, not results.
    """
    # KTC YoY
    ktc_yoy_log = 0.0
    has_prior = 0
    prior_end = prior.get("end_ktc") if prior else None
    if prior_end and prior_end > 0 and start_ktc > 0:
        ktc_yoy_log = float(np.clip(np.log(start_ktc / prior_end), -0.7, 0.7))
        has_prior = 1

    # Prior PPG
    prior_ppg = 0.0
    has_prior_ppg = 0
    if prior:
        pgp = prior.get("games_played", 0) or 0
        pfp = prior.get("fantasy_points", 0) or 0
        if pgp >= 4:
            prior_ppg = pfp / pgp
            has_prior_ppg = 1

    # Peak drawdown (simplified — use ktc_yoy as proxy)
    ktc_peak_drawdown = ktc_yoy_log

    return {
        "draft_pick": float(season.get("draft_pick") or 0),
        "years_remaining": float(season.get("years_remaining") or 0),
        "start_ktc_quartile": _get_ktc_quartile(start_ktc),
        "age_prime_distance": _age_prime_distance(age, position),
        "start_position_rank": float(season.get("start_position_rank") or 0),
        "start_ktc": start_ktc,
        "start_ktc_was_sentinel": 1 if start_ktc >= 9999 else 0,
        "years_exp": float(season.get("years_exp") or 0),
        "ktc_yoy_log": ktc_yoy_log,
        "ktc_peak_drawdown": ktc_peak_drawdown,
        "has_prior_season": has_prior,
        "prior_ppg": prior_ppg,
        "has_prior_ppg": has_prior_ppg,
        "apy_cap_pct": float(season.get("apy_cap_pct") or 0),
        "is_contract_year": float(season.get("is_contract_year") or 0),
        "apy_position_rank": float(season.get("apy_position_rank") or 0),
        "has_contract_data": 1 if season.get("apy_cap_pct") else 0,
        # v3 (prior season behavioral)
        "prior_weekly_fp_cv": float(prior.get("weekly_fp_cv") or 0) if prior else 0,
        "prior_boom_rate": float(prior.get("boom_rate") or 0) if prior else 0,
        "prior_bust_rate": float(prior.get("bust_rate") or 0) if prior else 0,
        "prior_snap_pct": float(prior.get("snap_pct") or 0) if prior else 0,
        "prior_ktc_volatility": float(prior.get("ktc_volatility") or 0) if prior else 0,
        "has_prior_behavioral": 1 if prior and (prior.get("games_played", 0) or 0) >= 4 else 0,
        # v4 (pre-season KTC trends)
        "ktc_30d_trend": float(season.get("ktc_30d_trend") or 0),
        "ktc_90d_trend": float(season.get("ktc_90d_trend") or 0),
        "momentum_ratio": float(season.get("momentum_ratio") or 0),
        "max_games_missed_streak": float(season.get("max_games_missed_streak") or 0),
        # Position-specific (prior season)
        "prior_passing_tds": float(prior.get("passing_tds") or 0) if prior else 0,
        "prior_interceptions": float(prior.get("interceptions") or 0) if prior else 0,
        "prior_carries": float(prior.get("carries") or 0) if prior else 0,
        "prior_red_zone_touches": float(prior.get("red_zone_touches") or 0) if prior else 0,
        "prior_targets": float(prior.get("targets") or 0) if prior else 0,
        "prior_red_zone_targets": float(prior.get("red_zone_targets") or 0) if prior else 0,
        "has_prior_position_stats": 1 if prior and (prior.get("games_played", 0) or 0) >= 4 else 0,
    }


class CompsIndex:
    """KNN index over historical player-seasons, weighted by model importances."""

    def __init__(self):
        self.nn: dict[str, NearestNeighbors] = {}
        self.scalers: dict[str, StandardScaler] = {}
        self.feature_names: dict[str, list[str]] = {}
        self.weights: dict[str, np.ndarray] = {}
        self.records: dict[str, list[dict]] = {}
        self.model_id: str = ""

    def build(
        self,
        model_bundle: dict,
        model_id: str,
        data_path: Path | None = None,
    ):
        """Build KNN indices weighted by the given model's feature importances."""
        self.model_id = model_id

        path = data_path or TRAINING_DATA_PATH
        zip_path = path.with_suffix(".zip") if path.suffix != ".zip" else path
        json_path = path.with_suffix(".json") if path.suffix != ".json" else path
        if zip_path.exists():
            with zipfile.ZipFile(zip_path) as zf:
                with zf.open(json_path.name) as f:
                    raw = json.load(f)
        else:
            with open(json_path) as f:
                raw = json.load(f)
        players = raw.get("players", [])

        # Pre-compute prior seasons
        prior_map: dict[tuple[str, int], dict] = {}
        for player in players:
            pid = player["player_id"]
            sorted_s = sorted(
                [s for s in player.get("seasons", []) if (s.get("years_exp") or 0) >= 0],
                key=lambda s: s["year"],
            )
            for i, s in enumerate(sorted_s):
                if i > 0 and (sorted_s[i-1].get("games_played", 0) or 0) >= 4:
                    prior_map[(pid, s["year"])] = sorted_s[i-1]

        # Get the feature names the model was trained with
        saved_features = model_bundle.get("feature_names", {})

        for position in ["QB", "RB", "WR", "TE"]:
            # Determine which features this model uses for this position
            if isinstance(saved_features, dict) and position in saved_features:
                model_features = saved_features[position]
            elif isinstance(saved_features, list):
                model_features = saved_features
            else:
                continue

            # Extract importances from the trained model
            pos_model = model_bundle.get("models", {}).get(position)
            if pos_model is None:
                continue

            importances = _extract_feature_importances(pos_model, len(model_features))

            # Remove current-season features from the model's feature list.
            # Comps match on pre-season profile only.
            _CURRENT_SEASON_FEATURES = {
                "games_played", "ppg", "games_played_so_far", "ppg_so_far",
                "weeks_missed", "weeks_missed_so_far", "ppg_yoy_log",
                "is_breakout_candidate",
            }
            keep = [(i, f) for i, f in enumerate(model_features) if f not in _CURRENT_SEASON_FEATURES]
            if len(keep) < len(model_features):
                kept_idx, model_features = zip(*keep)
                model_features = list(model_features)
                importances = importances[list(kept_idx)]

            # Add extra features that aren't in the model but matter for comps.
            # years_exp ensures rookies match rookies, not 3rd-year players.
            # start_position_rank differentiates players at similar KTC.
            _EXTRA_COMP_FEATURES = {
                "years_exp": 0.15,
                "start_position_rank": 0.08,
            }
            for fname, min_weight in _EXTRA_COMP_FEATURES.items():
                if fname not in model_features:
                    model_features = model_features + [fname]
                    importances = np.append(importances, min_weight)

            # Ensure start_ktc gets meaningful weight in comps regardless of
            # model split frequency. The model uses start_ktc_quartile for
            # splits, but comps need similar absolute KTC to be useful.
            _MIN_WEIGHTS = {"start_ktc": 0.12}
            for fname, min_w in _MIN_WEIGHTS.items():
                if fname in model_features:
                    idx = model_features.index(fname)
                    if importances[idx] < min_w:
                        importances[idx] = min_w

            importances = importances / importances.sum()

            # Build records and feature vectors
            records = []
            feature_rows = []

            for player in players:
                if player["position"] != position:
                    continue
                pid = player["player_id"]

                for season in player.get("seasons", []):
                    if (season.get("years_exp") or 0) < 0:
                        continue
                    start_ktc = season.get("start_ktc")
                    end_ktc = season.get("end_ktc")
                    gp = season.get("games_played", 0) or 0
                    fp = season.get("fantasy_points", 0) or 0
                    age = season.get("age")

                    if not start_ktc or start_ktc <= 0 or not end_ktc or gp < 1 or age is None:
                        continue

                    ppg = fp / gp
                    prior = prior_map.get((pid, season["year"]))

                    all_feats = _compute_all_features(season, prior, position, start_ktc, ppg, gp, age)

                    # Select only the features this model uses, in order
                    row = [all_feats.get(fname, 0.0) for fname in model_features]

                    records.append({
                        "player_id": pid,
                        "name": player["name"],
                        "position": position,
                        "year": season["year"],
                        "age": age,
                        "games_played": gp,
                        "ppg": round(ppg, 1),
                        "start_ktc": round(start_ktc, 1),
                        "end_ktc": round(end_ktc, 1),
                        "delta_ktc": round(end_ktc - start_ktc, 1),
                        "pct_change": round((end_ktc - start_ktc) / start_ktc * 100, 1),
                    })
                    feature_rows.append(row)

            if len(records) < 10:
                continue

            X = np.array(feature_rows)
            for col in range(X.shape[1]):
                mask = np.isnan(X[:, col])
                if mask.any():
                    X[mask, col] = float(np.nanmedian(X[:, col]))

            scaler = StandardScaler()
            X_scaled = scaler.fit_transform(X) * importances

            nn = NearestNeighbors(n_neighbors=min(30, len(records)), metric="euclidean")
            nn.fit(X_scaled)

            self.nn[position] = nn
            self.scalers[position] = scaler
            self.feature_names[position] = model_features
            self.weights[position] = importances
            self.records[position] = records

        total = sum(len(r) for r in self.records.values())
        logger.info(
            "CompsIndex built for model '%s': %d player-seasons, features: %s",
            model_id, total,
            {p: len(f) for p, f in self.feature_names.items()},
        )

    def find_comps(
        self,
        position: str,
        start_ktc: float,
        age: float,
        k: int = 10,
        exclude_player_id: str | None = None,
        # Legacy params accepted but not used for matching
        ppg: float = 0.0,
        games_played: int = 0,
        **kwargs,
    ) -> list[dict]:
        """Find the k most similar historical player-seasons.

        Matches on pre-season features only (KTC, age, prior stats, etc.).
        Current-season PPG/games are not used for matching.
        """
        if position not in self.nn:
            return []

        feature_names = self.feature_names[position]
        weights = self.weights[position]

        all_feats = {
            "start_ktc": start_ktc,
            "age_prime_distance": _age_prime_distance(age, position),
            "start_ktc_quartile": _get_ktc_quartile(start_ktc),
        }
        all_feats.update(kwargs)

        query = [float(all_feats.get(fname, 0.0) or 0.0) for fname in feature_names]
        query_arr = np.array([query])
        query_scaled = self.scalers[position].transform(query_arr) * weights

        n_fetch = min(k + 20, len(self.records[position]))
        distances, indices = self.nn[position].kneighbors(query_scaled, n_neighbors=n_fetch)

        results = []
        for dist, idx in zip(distances[0], indices[0]):
            record = self.records[position][idx]
            if exclude_player_id and record["player_id"] == exclude_player_id:
                continue
            results.append({
                **record,
                "similarity": round(float(1.0 / (1.0 + dist)), 4),
            })
            if len(results) >= k:
                break

        return results


# ── Cache: one CompsIndex per model_id ──────────────────────────────────

_cache: dict[str, CompsIndex] = {}


def get_comps_index(model_id: str | None = None) -> CompsIndex:
    """Get or build a CompsIndex for the given model iteration.

    Caches per model_id so switching models doesn't rebuild unnecessarily.
    """
    from app.services.model_registry import get_registry

    registry = get_registry()
    iteration = registry.get(model_id)
    mid = iteration.id

    if mid in _cache:
        return _cache[mid]

    idx = CompsIndex()
    idx.build(model_bundle=iteration.bundle, model_id=mid)
    _cache[mid] = idx
    return idx
