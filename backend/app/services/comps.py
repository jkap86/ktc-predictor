"""Find historical comparable players using the same features the model uses.

Builds per-position KNN indices over training data using the model's actual
feature set (position-specific, matching v4's layout). Returns the most
similar historical player-seasons with their actual outcomes.

Feature alignment: the comp finder computes the same engineered features
(KTC quartile, age-prime distance, breakout flag, prior-season KTC/PPG
trajectory, behavioral signals) that the training pipeline builds in
v1_hgb_baseline/data.py, so similarity reflects the model's view of
each player-season.
"""

import json
import logging
import zipfile
from pathlib import Path
from typing import Optional

import numpy as np
from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import StandardScaler

from app.config import TRAINING_DATA_PATH

logger = logging.getLogger(__name__)

# Prime age by position (matches model's predict.py)
_PRIME_AGE = {"QB": 27, "RB": 24, "WR": 26, "TE": 27}
_KTC_QUARTILE_BOUNDS = [1559, 3085, 4850]


def _get_ktc_quartile(ktc: float) -> int:
    if ktc < 1559: return 1
    elif ktc < 3085: return 2
    elif ktc < 4850: return 3
    return 4


def _age_prime_distance(age: float, position: str) -> float:
    return age - _PRIME_AGE.get(position, 26)

_comps_index: Optional["CompsIndex"] = None


class CompsIndex:
    """Pre-built KNN index over all historical player-seasons."""

    def __init__(self):
        self.nn: dict[str, NearestNeighbors] = {}
        self.scalers: dict[str, StandardScaler] = {}
        self.feature_names: dict[str, list[str]] = {}
        self.weights: dict[str, np.ndarray] = {}
        self.records: dict[str, list[dict]] = {}

    def _get_feature_spec(self, position: str) -> list[tuple[str, float]]:
        """Return (feature_name, weight) list for a position, aligned with the model."""
        # Core features (all positions) — match v1 core
        spec = [
            ("games_played", 1.0),
            ("ppg", 2.5),
            ("weeks_missed", 0.5),
            ("draft_pick", 0.5),
            ("start_ktc_quartile", 1.5),
            ("age_prime_distance", 2.0),
            # Base linear
            ("start_ktc", 3.0),
            # Prior-season KTC trajectory
            ("ktc_yoy_pct", 1.5),
            # Prior-season PPG
            ("prior_ppg", 1.0),
            # Current-season behavioral
            ("ktc_30d_trend", 1.5),
            ("ktc_volatility", 1.0),
            ("boom_rate", 1.0),
            ("bust_rate", 1.0),
            ("start_position_rank", 1.5),
            ("prior_year_fp", 1.0),
            ("snap_pct", 0.8),
        ]

        # v3 prior-behavioral (RB/WR only)
        if position in ("RB", "WR"):
            spec.extend([
                ("prior_weekly_fp_cv", 0.8),
                ("prior_boom_rate", 0.8),
                ("prior_bust_rate", 0.8),
                ("prior_snap_pct", 0.8),
            ])

        # v4 momentum (QB/WR only)
        if position in ("QB", "WR"):
            spec.append(("momentum_ratio", 1.0))
            spec.append(("max_games_missed_streak", 0.8))

        # Position-specific stats
        if position == "QB":
            spec.extend([("passing_tds", 1.5), ("interceptions", 1.0)])
        elif position == "RB":
            spec.extend([("carries", 1.5), ("red_zone_touches", 1.0)])
        elif position in ("WR", "TE"):
            spec.extend([("targets", 1.5), ("red_zone_targets", 1.0)])

        return spec

    def build(self, data_path: Path | None = None):
        """Load training data and build per-position KNN indices."""
        path = data_path or TRAINING_DATA_PATH
        with open(path) as f:
            raw = json.load(f)

        players = raw.get("players", [])

        # Pre-compute prior-season data for KTC trajectory and prior behavioral
        prior_data: dict[tuple[str, int], dict] = {}  # (pid, year) -> prior season
        for player in players:
            pid = player["player_id"]
            sorted_seasons = sorted(
                [s for s in player.get("seasons", []) if (s.get("years_exp") or 0) >= 0],
                key=lambda s: s["year"],
            )
            for i, season in enumerate(sorted_seasons):
                if i > 0:
                    prior = sorted_seasons[i - 1]
                    if (prior.get("games_played", 0) or 0) >= 4:
                        prior_data[(pid, season["year"])] = prior

        for position in ["QB", "RB", "WR", "TE"]:
            feat_spec = self._get_feature_spec(position)
            feature_names = [f[0] for f in feat_spec]
            weights = np.array([f[1] for f in feat_spec])

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

                    if not start_ktc or start_ktc <= 0 or not end_ktc:
                        continue
                    if gp < 1 or age is None:
                        continue

                    ppg = fp / gp
                    prior = prior_data.get((pid, season["year"]))

                    # Compute KTC YoY (simplified — not log, just pct for KNN)
                    ktc_yoy_pct = 0.0
                    if prior:
                        prior_end = prior.get("end_ktc")
                        if prior_end and prior_end > 0:
                            ktc_yoy_pct = (start_ktc - prior_end) / prior_end

                    prior_ppg = 0.0
                    if prior:
                        pgp = prior.get("games_played", 0) or 0
                        pfp = prior.get("fantasy_points", 0) or 0
                        if pgp > 0:
                            prior_ppg = pfp / pgp

                    # Build feature vector
                    feat_map = {
                        "games_played": gp,
                        "ppg": ppg,
                        "weeks_missed": (season.get("weekly_stats", [{}])[-1].get("week", gp) - gp) if season.get("weekly_stats") else 0,
                        "draft_pick": float(season.get("draft_pick") or 0),
                        "start_ktc_quartile": _get_ktc_quartile(start_ktc),
                        "age_prime_distance": _age_prime_distance(age, position),
                        "start_ktc": start_ktc,
                        "ktc_yoy_pct": ktc_yoy_pct,
                        "prior_ppg": prior_ppg,
                        "ktc_30d_trend": float(season.get("ktc_30d_trend") or 0),
                        "ktc_volatility": float(season.get("ktc_volatility") or 0),
                        "boom_rate": float(season.get("boom_rate") or 0),
                        "bust_rate": float(season.get("bust_rate") or 0),
                        "start_position_rank": float(season.get("start_position_rank") or 0),
                        "prior_year_fp": float(season.get("prior_year_fp") or 0),
                        "snap_pct": float(season.get("snap_pct") or 0),
                        # v3 prior-behavioral (from prior season)
                        "prior_weekly_fp_cv": float(prior.get("weekly_fp_cv") or 0) if prior else 0,
                        "prior_boom_rate": float(prior.get("boom_rate") or 0) if prior else 0,
                        "prior_bust_rate": float(prior.get("bust_rate") or 0) if prior else 0,
                        "prior_snap_pct": float(prior.get("snap_pct") or 0) if prior else 0,
                        # v4 momentum
                        "momentum_ratio": float(season.get("momentum_ratio") or 0),
                        "max_games_missed_streak": float(season.get("max_games_missed_streak") or 0),
                        # Position-specific
                        "passing_tds": float(season.get("passing_tds") or 0),
                        "interceptions": float(season.get("interceptions") or 0),
                        "carries": float(season.get("carries") or 0),
                        "red_zone_touches": float(season.get("red_zone_touches") or 0),
                        "targets": float(season.get("targets") or 0),
                        "red_zone_targets": float(season.get("red_zone_targets") or 0),
                    }

                    row = [feat_map.get(fname, 0.0) for fname in feature_names]

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
        **kwargs,
    ) -> list[dict]:
        """Find the k most similar historical player-seasons.

        Pass any model features as kwargs (e.g. ktc_30d_trend=0.05,
        boom_rate=0.3, prior_ppg=15.0). Unknown features are ignored.
        Missing features default to 0.
        """
        if position not in self.nn:
            return []

        feature_names = self.feature_names[position]
        weights = self.weights[position]

        # Compute engineered features from raw inputs
        computed = {
            "start_ktc": start_ktc,
            "ppg": ppg,
            "age_prime_distance": _age_prime_distance(age, position),
            "start_ktc_quartile": _get_ktc_quartile(start_ktc),
            "games_played": games_played,
        }
        computed.update(kwargs)

        query = [float(computed.get(fname, 0.0) or 0.0) for fname in feature_names]
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
