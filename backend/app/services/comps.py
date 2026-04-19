"""Find historical comparable players based on model input features.

Given a player's current profile (position, start_ktc, ppg, gp, age),
finds the most similar player-seasons from prior years in the training
data and returns their actual outcomes.
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

# Feature weights — higher = more important in similarity matching
# start_ktc and ppg matter most for dynasty value trajectory
_FEATURE_WEIGHTS = {
    "start_ktc": 2.0,
    "ppg": 2.0,
    "age": 1.5,
    "games_played": 1.0,
}

_comps_index: Optional["CompsIndex"] = None


class CompsIndex:
    """Pre-built KNN index over all historical player-seasons."""

    def __init__(self):
        self.data: list[dict] = []  # raw season records
        self.nn: dict[str, NearestNeighbors] = {}  # position -> fitted NN
        self.scalers: dict[str, StandardScaler] = {}
        self.features: dict[str, np.ndarray] = {}
        self.records: dict[str, list[dict]] = {}  # position -> list of season dicts

    def build(self, data_path: Path | None = None):
        """Load training data and build per-position KNN indices."""
        path = data_path or TRAINING_DATA_PATH
        with open(path) as f:
            raw = json.load(f)

        players = raw.get("players", [])
        feature_cols = ["start_ktc", "ppg", "age", "games_played"]
        weights = np.array([_FEATURE_WEIGHTS[c] for c in feature_cols])

        for position in ["QB", "RB", "WR", "TE"]:
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
                    feature_rows.append([start_ktc, ppg, age, gp])

            if len(records) < 10:
                continue

            X = np.array(feature_rows)
            scaler = StandardScaler()
            X_scaled = scaler.fit_transform(X) * weights

            nn = NearestNeighbors(n_neighbors=min(20, len(records)), metric="euclidean")
            nn.fit(X_scaled)

            self.nn[position] = nn
            self.scalers[position] = scaler
            self.features[position] = X
            self.records[position] = records

        total = sum(len(r) for r in self.records.values())
        logger.info("CompsIndex built: %d player-seasons across %d positions", total, len(self.nn))

    def find_comps(
        self,
        position: str,
        start_ktc: float,
        ppg: float,
        age: float,
        games_played: int,
        k: int = 10,
        exclude_player_id: str | None = None,
    ) -> list[dict]:
        """Find the k most similar historical player-seasons.

        Returns list of dicts sorted by similarity (closest first), each with
        player info, season stats, and actual outcome.
        """
        if position not in self.nn:
            return []

        weights = np.array([
            _FEATURE_WEIGHTS["start_ktc"],
            _FEATURE_WEIGHTS["ppg"],
            _FEATURE_WEIGHTS["age"],
            _FEATURE_WEIGHTS["games_played"],
        ])

        query = np.array([[start_ktc, ppg, age, games_played]])
        query_scaled = self.scalers[position].transform(query) * weights

        # Fetch extra neighbors in case we need to exclude the player
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
