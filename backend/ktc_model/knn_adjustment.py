"""KNN-based adjustment layer for elite tier predictions.

Elite players (6k+ KTC) are systematically under-predicted because the model
has learned historical mean-reversion too strongly. This module provides a
KNN-based correction that finds similar historical players and blends their
actual outcomes with the model prediction.
"""

import numpy as np
from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import StandardScaler


# Feature flag for contract features in KNN (must match train.py)
USE_CONTRACT_FEATURES_IN_KNN = True

# Position-specific blend weight multipliers
# QB has more stable trajectories, so KNN should be trusted more
# RB has high variance, so KNN helps less (but still useful early-season)
POSITION_BLEND_MULTIPLIERS = {
    "QB": 1.5,  # 50% more KNN influence for QB
    "RB": 1.0,  # Default for RB (variance too high for stronger reliance)
    "WR": 1.2,  # Slight boost for WR
    "TE": 1.3,  # Moderate boost for TE (stable like QB)
}


class EliteKNNAdjuster:
    """KNN-based adjustment for elite tier predictions.

    For players in the elite tier (default 6k+ KTC), finds similar historical
    players and blends their actual outcomes with the model prediction. This
    reduces the systematic under-prediction of elite risers.

    Attributes
    ----------
    k : int
        Number of neighbors to use for adjustment.
    elite_threshold : float
        KTC threshold above which KNN adjustment is applied.
    blend_weight : float
        Base weight for KNN contribution (0-1). Actual weight is modulated
        by neighbor distance.
    distance_decay : float
        Controls how quickly blend weight decreases with neighbor distance.
    """

    def __init__(
        self,
        k: int = 10,
        elite_threshold: float = 6000,
        blend_weight: float = 0.3,
        distance_decay: float = 2.0,
    ):
        self.k = k
        self.elite_threshold = elite_threshold
        self.blend_weight = blend_weight
        self.distance_decay = distance_decay

        # Per-position data (populated by fit())
        self.indices: dict = {}  # position -> NearestNeighbors
        self.scalers: dict = {}  # position -> StandardScaler
        self.outcomes: dict = {}  # position -> array of log_ratios
        self.features: dict = {}  # position -> array of raw features

    def fit(self, df, position: str) -> "EliteKNNAdjuster":
        """Build KNN index from training data for a position.

        Parameters
        ----------
        df : pd.DataFrame
            Training data with columns: position, start_ktc, age_prime_distance,
            ppg_so_far, games_played_so_far, log_ratio.
            Optionally includes contract features: apy_cap_pct, is_contract_year.
        position : str
            Position to build index for (QB, RB, WR, TE).

        Returns
        -------
        self
        """
        # Filter to elite tier for this position
        mask = (df["position"] == position) & (df["start_ktc"] >= self.elite_threshold)
        elite = df[mask].copy()

        if len(elite) < self.k:
            # Not enough samples for KNN
            return self

        # Base features for similarity matching
        feature_cols = ["age_prime_distance", "ppg_so_far", "start_ktc", "games_played_so_far"]

        # Add contract features if available and enabled
        has_contract_features = (
            USE_CONTRACT_FEATURES_IN_KNN
            and "apy_cap_pct" in elite.columns
            and elite["apy_cap_pct"].notna().sum() > self.k
        )
        if has_contract_features:
            feature_cols.extend(["apy_cap_pct", "is_contract_year"])

        self.feature_cols = feature_cols  # Store for use in adjust()

        X = elite[feature_cols].values
        y = elite["log_ratio"].values

        # Handle any NaN values
        nan_mask = ~np.isnan(X).any(axis=1) & ~np.isnan(y)
        X = X[nan_mask]
        y = y[nan_mask]

        if len(X) < self.k:
            return self

        # Scale features
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)

        # Build KNN index
        nn = NearestNeighbors(n_neighbors=self.k, metric="euclidean")
        nn.fit(X_scaled)

        self.indices[position] = nn
        self.scalers[position] = scaler
        self.outcomes[position] = y
        self.features[position] = X

        return self

    def adjust(
        self,
        position: str,
        model_log_ratio: float,
        age_prime_dist: float,
        ppg: float,
        start_ktc: float,
        gp: float,
        apy_cap_pct: float | None = None,
        is_contract_year: float | None = None,
    ) -> float:
        """Adjust model prediction using KNN neighbors.

        Blends the model's prediction with the distance-weighted mean outcome of
        similar historical players. The blend weight is modulated by games played,
        position stability, and elite tier membership.

        Parameters
        ----------
        position : str
            Player position (QB, RB, WR, TE).
        model_log_ratio : float
            Raw log_ratio prediction from the model.
        age_prime_dist : float
            Player's age distance from positional prime.
        ppg : float
            Points per game so far this season.
        start_ktc : float
            Start of season KTC value.
        gp : float
            Games played so far.
        apy_cap_pct : float or None
            APY as percentage of salary cap (contract feature).
        is_contract_year : float or None
            1 if in final year of contract, 0 otherwise.

        Returns
        -------
        float
            Adjusted log_ratio prediction.
        """
        # QB gets a lower threshold to catch 5.5k-6k "breakout zone" players
        # This tier has the worst riser under-prediction for QB
        effective_threshold = 5500 if position == "QB" else self.elite_threshold

        # No adjustment for non-elite tier
        if start_ktc < effective_threshold:
            return model_log_ratio

        # No KNN data for this position
        if position not in self.indices:
            return model_log_ratio

        # Build query vector - base features
        query_features = [age_prime_dist, ppg, start_ktc, gp]

        # Add contract features if KNN was trained with them
        expected_n_features = self.features[position].shape[1]
        if expected_n_features > 4:
            # KNN was trained with contract features
            if apy_cap_pct is not None and is_contract_year is not None:
                query_features.extend([apy_cap_pct, is_contract_year])
            else:
                # Use defaults (average values) if contract data not available
                query_features.extend([0.05, 0])  # ~median apy_cap_pct, not contract year

        X_query = np.array([query_features])

        # Handle NaN in query
        if np.isnan(X_query).any():
            return model_log_ratio

        # Scale and find neighbors
        X_scaled = self.scalers[position].transform(X_query)
        distances, indices = self.indices[position].kneighbors(X_scaled)

        # Get neighbor outcomes (actual log_ratios)
        neighbor_outcomes = self.outcomes[position][indices[0]]
        neighbor_distances = distances[0]

        # Distance-weighted averaging instead of simple mean
        # Closer neighbors get more influence on the prediction
        eps = 1e-6  # Prevent division by zero
        inv_distances = 1.0 / (neighbor_distances + eps)
        weights = inv_distances / inv_distances.sum()
        knn_weighted_mean = np.sum(neighbor_outcomes * weights)

        # Games-played adaptive blending
        # Early-season predictions (low GP) should trust historical neighbors more
        # Late-season predictions (high GP) should trust the model more
        if gp <= 3:
            gp_weight_multiplier = 2.0  # Double weight for early season
        elif gp <= 8:
            gp_weight_multiplier = 1.5  # 1.5x for mid-season
        else:
            gp_weight_multiplier = 1.0  # Normal weight for late season

        # Position-specific blend weight multiplier
        # QB trajectories are more stable/predictable, so trust KNN more
        position_multiplier = POSITION_BLEND_MULTIPLIERS.get(position, 1.0)

        # Elite tier boost: 6k+ players benefit more from KNN due to under-prediction
        elite_multiplier = 1.5 if start_ktc >= 6000 else 1.0

        # Modulate blend weight by neighbor distance, games played, position, AND elite tier
        # Closer neighbors = higher effective weight
        avg_distance = np.mean(neighbor_distances)
        base_weight = self.blend_weight * gp_weight_multiplier * position_multiplier * elite_multiplier
        effective_weight = base_weight * np.exp(-avg_distance / self.distance_decay)

        # Cap effective weight - higher cap for elite tier
        if start_ktc >= 6000:
            max_weight = 0.8  # Higher cap for elite tier
        elif position == "QB":
            max_weight = 0.7  # QB is more trustworthy
        else:
            max_weight = 0.6
        effective_weight = min(max_weight, effective_weight)

        # Blend model prediction with distance-weighted KNN prediction
        adjusted = (1 - effective_weight) * model_log_ratio + effective_weight * knn_weighted_mean

        return adjusted

    def get_neighbors(
        self,
        position: str,
        age_prime_dist: float,
        ppg: float,
        start_ktc: float,
        gp: float,
        apy_cap_pct: float | None = None,
        is_contract_year: float | None = None,
    ) -> list[dict] | None:
        """Get the k nearest neighbors for debugging/inspection.

        Returns
        -------
        list[dict] or None
            List of neighbor info dicts, or None if not applicable.
        """
        # QB gets a lower threshold (matching adjust() logic)
        effective_threshold = 5500 if position == "QB" else self.elite_threshold

        if start_ktc < effective_threshold:
            return None

        if position not in self.indices:
            return None

        # Build query vector - base features
        query_features = [age_prime_dist, ppg, start_ktc, gp]

        # Add contract features if KNN was trained with them
        expected_n_features = self.features[position].shape[1]
        if expected_n_features > 4:
            if apy_cap_pct is not None and is_contract_year is not None:
                query_features.extend([apy_cap_pct, is_contract_year])
            else:
                query_features.extend([0.05, 0])

        X_query = np.array([query_features])
        if np.isnan(X_query).any():
            return None

        X_scaled = self.scalers[position].transform(X_query)
        distances, indices = self.indices[position].kneighbors(X_scaled)

        neighbors = []
        for i, (dist, idx) in enumerate(zip(distances[0], indices[0])):
            features = self.features[position][idx]
            neighbor_info = {
                "rank": i + 1,
                "distance": float(dist),
                "age_prime_dist": float(features[0]),
                "ppg": float(features[1]),
                "start_ktc": float(features[2]),
                "games_played": float(features[3]),
                "actual_log_ratio": float(self.outcomes[position][idx]),
            }
            # Add contract features if present
            if len(features) > 4:
                neighbor_info["apy_cap_pct"] = float(features[4])
                neighbor_info["is_contract_year"] = int(features[5])
            neighbors.append(neighbor_info)

        return neighbors

    def __repr__(self) -> str:
        positions = list(self.indices.keys())
        counts = {p: len(self.outcomes.get(p, [])) for p in positions}
        return (
            f"EliteKNNAdjuster(k={self.k}, threshold={self.elite_threshold}, "
            f"positions={counts})"
        )
