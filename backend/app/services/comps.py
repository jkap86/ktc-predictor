"""Find historical comparable players using the same features the model uses.

Builds per-position KNN indices over training data using the model's actual
feature set (position-specific, matching v4's layout). Returns the most
similar historical player-seasons with their actual outcomes.
"""

import json
import logging
from pathlib import Path
from typing import Optional

import numpy as np
from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import StandardScaler

from app.config import TRAINING_DATA_PATH

logger = logging.getLogger(__name__)

# Features used for comp matching, aligned with the model's inputs.
# Weights reflect feature importance: higher = more influence on similarity.
# Grouped: core inputs, KTC trajectory, prior performance, behavioral.
_COMP_FEATURES = {
    # Core model inputs (all positions)
    "start_ktc":         3.0,   # Current dynasty value — strongest signal
    "ppg":               2.5,   # Per-game production
    "age":               2.0,   # Career arc position
    "games_played":      1.0,   # Sample size / health
    "start_position_rank": 1.5, # Positional tier
    # KTC trajectory
    "ktc_30d_trend":     1.5,   # Recent momentum
    "ktc_volatility":    1.0,   # Value stability
    # Prior-season performance
    "prior_year_fp":     1.5,   # Last year's total FP
    "boom_rate":         1.0,   # Upside frequency
    "bust_rate":         1.0,   # Downside frequency
}

# Position-specific extra features (only added when available)
_POSITION_EXTRA_FEATURES = {
    "QB": {"passing_tds": 1.5, "interceptions": 1.0},
    "RB": {"carries": 1.5, "red_zone_touches": 1.0},
    "WR": {"targets": 1.5, "red_zone_targets": 1.0},
    "TE": {"targets": 1.5, "red_zone_targets": 1.0},
}

_comps_index: Optional["CompsIndex"] = None


class CompsIndex:
    """Pre-built KNN index over all historical player-seasons."""

    def __init__(self):
        self.nn: dict[str, NearestNeighbors] = {}
        self.scalers: dict[str, StandardScaler] = {}
        self.feature_names: dict[str, list[str]] = {}
        self.weights: dict[str, np.ndarray] = {}
        self.records: dict[str, list[dict]] = {}

    def build(self, data_path: Path | None = None):
        """Load training data and build per-position KNN indices."""
        path = data_path or TRAINING_DATA_PATH
        with open(path) as f:
            raw = json.load(f)

        players = raw.get("players", [])

        for position in ["QB", "RB", "WR", "TE"]:
            # Build feature list for this position
            feat_weights = dict(_COMP_FEATURES)
            feat_weights.update(_POSITION_EXTRA_FEATURES.get(position, {}))
            feature_names = list(feat_weights.keys())
            weights = np.array([feat_weights[f] for f in feature_names])

            records = []
            feature_rows = []

            for player in players:
                if player["position"] != position:
                    continue

                for season in player.get("seasons", []):
                    if (season.get("years_exp") or 0) < 0:
                        continue
                    start_ktc = season.get("start_ktc")
                    end_ktc = season.get("end_ktc")
                    gp = season.get("games_played", 0) or 0
                    fp = season.get("fantasy_points", 0) or 0
                    age = season.get("age")

                    if not start_ktc or start_ktc <= 0 or not end_ktc:
                        continue
                    if gp < 1 or age is None:
                        continue

                    ppg = fp / gp

                    # Extract all features, using 0 for missing
                    row = []
                    for fname in feature_names:
                        if fname == "ppg":
                            row.append(ppg)
                        elif fname == "games_played":
                            row.append(gp)
                        elif fname == "age":
                            row.append(age)
                        else:
                            row.append(float(season.get(fname) or 0))

                    # Skip if too many NaN/zero core features
                    if row[0] <= 0:  # start_ktc
                        continue

                    records.append({
                        "player_id": player["player_id"],
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
            # Replace NaN with column median before scaling
            for col in range(X.shape[1]):
                mask = np.isnan(X[:, col])
                if mask.any():
                    median = np.nanmedian(X[:, col])
                    X[mask, col] = median

            scaler = StandardScaler()
            X_scaled = scaler.fit_transform(X) * weights

            nn = NearestNeighbors(n_neighbors=min(30, len(records)), metric="euclidean")
            nn.fit(X_scaled)

            self.nn[position] = nn
            self.scalers[position] = scaler
            self.feature_names[position] = feature_names
            self.weights[position] = weights
            self.records[position] = records

        total = sum(len(r) for r in self.records.values())
        logger.info(
            "CompsIndex built: %d player-seasons across %d positions, features per pos: %s",
            total,
            len(self.nn),
            {p: len(f) for p, f in self.feature_names.items()},
        )

    def find_comps(
        self,
        position: str,
        start_ktc: float,
        ppg: float,
        age: float,
        games_played: int,
        k: int = 10,
        exclude_player_id: str | None = None,
        # Additional features for richer matching (pass what's available)
        start_position_rank: int | None = None,
        ktc_30d_trend: float | None = None,
        ktc_volatility: float | None = None,
        prior_year_fp: float | None = None,
        boom_rate: float | None = None,
        bust_rate: float | None = None,
        # Position-specific
        passing_tds: float | None = None,
        interceptions: float | None = None,
        carries: float | None = None,
        red_zone_touches: float | None = None,
        targets: float | None = None,
        red_zone_targets: float | None = None,
    ) -> list[dict]:
        """Find the k most similar historical player-seasons."""
        if position not in self.nn:
            return []

        feature_names = self.feature_names[position]
        weights = self.weights[position]

        # Build query vector matching the feature order
        local_vars = {
            "start_ktc": start_ktc,
            "ppg": ppg,
            "age": age,
            "games_played": games_played,
            "start_position_rank": start_position_rank,
            "ktc_30d_trend": ktc_30d_trend,
            "ktc_volatility": ktc_volatility,
            "prior_year_fp": prior_year_fp,
            "boom_rate": boom_rate,
            "bust_rate": bust_rate,
            "passing_tds": passing_tds,
            "interceptions": interceptions,
            "carries": carries,
            "red_zone_touches": red_zone_touches,
            "targets": targets,
            "red_zone_targets": red_zone_targets,
        }

        query = []
        for fname in feature_names:
            val = local_vars.get(fname)
            query.append(float(val) if val is not None else 0.0)

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


def get_comps_index() -> CompsIndex:
    """Get or build the global comps index singleton."""
    global _comps_index
    if _comps_index is None:
        _comps_index = CompsIndex()
        _comps_index.build()
    return _comps_index
