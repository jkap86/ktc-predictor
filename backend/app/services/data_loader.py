import json
import numpy as np
import pandas as pd
from pathlib import Path
from typing import Optional

from app.config import TRAINING_DATA_PATH


def get_age_bracket(age: int, position: str) -> str:
    """Get age bracket based on position-specific aging curves."""
    if position == "RB":
        if age <= 24:
            return "young"
        elif age <= 27:
            return "prime"
        else:
            return "declining"
    elif position == "QB":
        if age <= 26:
            return "young"
        elif age <= 33:
            return "prime"
        else:
            return "declining"
    else:  # WR, TE
        if age <= 25:
            return "young"
        elif age <= 29:
            return "prime"
        else:
            return "declining"


def calculate_derived_features(season: dict) -> dict:
    """Calculate derived features from weekly data."""
    weekly_stats = season.get("weekly_stats", [])
    weekly_ktc = season.get("weekly_ktc", [])

    # Fantasy points features
    weekly_fp = [w["fantasy_points"] for w in weekly_stats]
    active_fp = [fp for fp in weekly_fp if fp > 0]

    fp_std_dev = float(np.std(active_fp)) if len(active_fp) > 1 else 0.0
    fp_consistency = len(active_fp) / 18.0 if weekly_stats else 0.0
    fp_max_week = max(active_fp) if active_fp else 0.0

    mid = len(weekly_fp) // 2
    first_half = sum(weekly_fp[:mid]) if mid > 0 else 0
    second_half = sum(weekly_fp[mid:])
    fp_second_half_ratio = (second_half / first_half) if first_half > 0 else 1.0

    # KTC features
    ktc_values = [w["ktc"] for w in weekly_ktc if w["ktc"] > 0]
    ktc_in_season_volatility = float(np.std(ktc_values)) if len(ktc_values) > 1 else 0.0

    if len(ktc_values) >= 2 and ktc_values[0] > 0:
        ktc_season_trend = (ktc_values[-1] - ktc_values[0]) / ktc_values[0]
        ktc_max_swing = (max(ktc_values) - min(ktc_values)) / ktc_values[0]
    else:
        ktc_season_trend = 0.0
        ktc_max_swing = 0.0

    # Snap percentage features
    weekly_snap_pct = [w.get("snap_pct", 0) for w in weekly_stats if w.get("snap_pct")]
    if len(weekly_snap_pct) >= 2:
        snap_pct_avg = float(np.mean(weekly_snap_pct))
        # Trend: change from first to last (normalized by 100 since snap_pct is 0-100)
        snap_pct_trend = (weekly_snap_pct[-1] - weekly_snap_pct[0]) / 100.0 if weekly_snap_pct[0] > 0 else 0.0
    else:
        snap_pct_avg = 0.0
        snap_pct_trend = 0.0

    return {
        "fp_std_dev": fp_std_dev,
        "fp_consistency": fp_consistency,
        "fp_max_week": fp_max_week,
        "fp_second_half_ratio": fp_second_half_ratio,
        "ktc_in_season_volatility": ktc_in_season_volatility,
        "ktc_season_trend": ktc_season_trend,
        "ktc_max_swing": ktc_max_swing,
        "snap_pct_avg": snap_pct_avg,
        "snap_pct_trend": snap_pct_trend,
    }


class DataLoader:
    def __init__(self, data_path: Optional[Path] = None):
        self.data_path = data_path or TRAINING_DATA_PATH
        self._raw_data: Optional[dict] = None
        self._players_df: Optional[pd.DataFrame] = None
        self._seasons_df: Optional[pd.DataFrame] = None

    def load(self) -> dict:
        """Load raw JSON data."""
        if self._raw_data is None:
            with open(self.data_path, "r") as f:
                self._raw_data = json.load(f)
        return self._raw_data

    def get_players(self) -> list[dict]:
        """Get list of all players."""
        data = self.load()
        return data.get("players", [])

    def get_player_by_id(self, player_id: str) -> Optional[dict]:
        """Get a single player by ID."""
        for player in self.get_players():
            if player["player_id"] == player_id:
                return player
        return None

    def search_players(
        self,
        query: str = "",
        position: Optional[str] = None,
        limit: int = 50,
        sort_by: str = "name",
        sort_order: str = "asc",
    ) -> list[dict]:
        """Search players by name and optionally filter by position.

        Args:
            query: Search query for player name
            position: Filter by position (QB, RB, WR, TE)
            limit: Maximum number of results
            sort_by: Sort by 'name' or 'ktc'
            sort_order: 'asc' or 'desc'
        """
        players = self.get_players()
        results = []

        for player in players:
            if position and player["position"] != position:
                continue
            if query.lower() in player["name"].lower():
                # Get latest season info
                seasons = player.get("seasons", [])
                latest_ktc = None
                latest_year = None
                if seasons:
                    latest = max(seasons, key=lambda s: s["year"])
                    latest_ktc = latest.get("end_ktc") or latest.get("start_ktc")
                    latest_year = latest["year"]

                results.append(
                    {
                        "player_id": player["player_id"],
                        "name": player["name"],
                        "position": player["position"],
                        "latest_ktc": latest_ktc,
                        "latest_year": latest_year,
                    }
                )

        # Sort results
        if sort_by == "ktc":
            # Sort by latest_ktc (nulls last)
            results.sort(
                key=lambda x: (x["latest_ktc"] is None, -(x["latest_ktc"] or 0)),
                reverse=(sort_order == "asc"),
            )
        else:  # sort_by == "name"
            results.sort(
                key=lambda x: x["name"].lower(),
                reverse=(sort_order == "desc"),
            )

        return results[:limit]

    def get_training_dataframe(self) -> pd.DataFrame:
        """Convert data to DataFrame for ML training."""
        if self._seasons_df is not None:
            return self._seasons_df

        players = self.get_players()
        rows = []

        for player in players:
            for season in player.get("seasons", []):
                # Calculate derived features from weekly data
                derived = calculate_derived_features(season)

                row = {
                    "player_id": player["player_id"],
                    "name": player["name"],
                    "position": player["position"],
                    "year": season["year"],
                    "age": season["age"],
                    "years_exp": season["years_exp"],
                    "start_ktc": season["start_ktc"],
                    "end_ktc": season["end_ktc"],
                    "fantasy_points": season["fantasy_points"],
                    "games_played": season["games_played"],
                    "ktc_30d_trend": season.get("ktc_30d_trend"),
                    "ktc_90d_trend": season.get("ktc_90d_trend"),
                    "ktc_volatility": season.get("ktc_volatility", 0),
                    "prior_year_fp": season.get("prior_year_fp", 0),
                    "prior_year_games": season.get("prior_year_games", 0),
                    "fp_change_yoy": season.get("fp_change_yoy", 0),
                    "start_position_rank": season.get("start_position_rank", 999),
                    # Derived features from weekly data
                    **derived,
                }
                rows.append(row)

        self._seasons_df = pd.DataFrame(rows)
        return self._seasons_df

    def get_feature_matrix(
        self, prior_predictions: Optional[dict] = None
    ) -> tuple[pd.DataFrame, pd.Series]:
        """Prepare features and target for ML model.

        Creates training pairs where:
        - Features come from season N
        - Target is end_ktc from season N+1

        This prevents data leakage by ensuring we only use
        past data to predict future outcomes.

        Args:
            prior_predictions: Optional dict mapping (player_id, year) -> predicted_ktc
                              If provided, adds prior_predicted_ktc feature
        """
        df = self.get_training_dataframe()

        # Sort by player and year
        df = df.sort_values(["player_id", "year"])

        # Pre-compute position/age averages for normalization
        position_age_fp_avg = df.groupby(["position", "age"])["fantasy_points"].mean()
        global_fp_avg = df["fantasy_points"].mean()

        training_rows = []

        # Group by player to create year-over-year pairs
        for player_id, player_df in df.groupby("player_id"):
            player_df = player_df.sort_values("year")
            seasons = player_df.to_dict("records")

            # Create pairs: season N features -> season N+1 target
            for i in range(len(seasons) - 1):
                current_season = seasons[i]
                next_season = seasons[i + 1]

                # Skip if years aren't consecutive
                if next_season["year"] != current_season["year"] + 1:
                    continue

                # Skip invalid KTC values
                if current_season["end_ktc"] <= 0 or next_season["end_ktc"] <= 0:
                    continue

                # Look up prior prediction if available
                prior_predicted_ktc = 0.0
                if prior_predictions:
                    key = (player_id, current_season["year"])
                    prior_predicted_ktc = prior_predictions.get(key, 0.0)

                # Calculate age for next season and age bracket
                next_age = current_season["age"] + 1
                age_bracket = get_age_bracket(next_age, current_season["position"])

                # Calculate position-normalized fantasy points
                pos = current_season["position"]
                age = current_season["age"]
                fp = current_season["fantasy_points"]
                try:
                    avg_fp = position_age_fp_avg.loc[(pos, age)]
                except KeyError:
                    avg_fp = global_fp_avg
                fp_vs_position_avg = fp / avg_fp if avg_fp > 0 else 1.0

                # Calculate feature interactions
                games_played = current_season["games_played"]
                fp_consistency = current_season.get("fp_consistency", 0)
                current_ktc = current_season["end_ktc"]
                ktc_volatility = current_season.get("ktc_in_season_volatility", 0)
                ktc_trend = current_season.get("ktc_season_trend", 0)

                row = {
                    # Features from current season (N)
                    "current_ktc": current_ktc,
                    "age": next_age,
                    "years_exp": current_season["years_exp"] + 1,
                    "fantasy_points": current_season["fantasy_points"],
                    "games_played": games_played,
                    "games_missed": 17 - games_played,
                    "fp_vs_position_avg": fp_vs_position_avg,
                    "position": pos,
                    "age_bracket": age_bracket,
                    # Derived features from current season
                    "fp_std_dev": current_season.get("fp_std_dev", 0),
                    "fp_consistency": fp_consistency,
                    "fp_max_week": current_season.get("fp_max_week", 0),
                    "fp_second_half_ratio": current_season.get(
                        "fp_second_half_ratio", 1
                    ),
                    "ktc_in_season_volatility": ktc_volatility,
                    "ktc_season_trend": ktc_trend,
                    "ktc_max_swing": current_season.get("ktc_max_swing", 0),
                    "snap_pct_avg": current_season.get("snap_pct_avg", 0),
                    "snap_pct_trend": current_season.get("snap_pct_trend", 0),
                    "prior_predicted_ktc": prior_predicted_ktc,
                    # Feature interactions
                    "age_x_is_rb": next_age * (1 if pos == "RB" else 0),
                    "age_x_is_qb": next_age * (1 if pos == "QB" else 0),
                    "games_x_consistency": games_played * fp_consistency,
                    "ktc_x_volatility": current_ktc * ktc_volatility / 10000,
                    "ktc_x_trend": current_ktc * ktc_trend / 10000,
                    # Target: next season's end KTC
                    "target_ktc": next_season["end_ktc"],
                }
                training_rows.append(row)

        train_df = pd.DataFrame(training_rows)

        # One-hot encode position
        position_dummies = pd.get_dummies(train_df["position"], prefix="pos")
        train_df = pd.concat([train_df, position_dummies], axis=1)

        # One-hot encode age bracket
        age_bracket_dummies = pd.get_dummies(train_df["age_bracket"], prefix="age")
        train_df = pd.concat([train_df, age_bracket_dummies], axis=1)

        # Feature columns
        feature_cols = [
            "current_ktc",
            "age",
            "years_exp",
            "fantasy_points",
            "games_played",
            "games_missed",
            "fp_vs_position_avg",
            "fp_std_dev",
            "fp_consistency",
            "fp_max_week",
            "fp_second_half_ratio",
            "ktc_in_season_volatility",
            "ktc_season_trend",
            "ktc_max_swing",
            "snap_pct_avg",
            "snap_pct_trend",
            "prior_predicted_ktc",
            # Feature interactions
            "age_x_is_rb",
            "age_x_is_qb",
            "games_x_consistency",
            "ktc_x_volatility",
            "ktc_x_trend",
        ] + [c for c in train_df.columns if c.startswith("pos_")] + [
            c for c in train_df.columns if c.startswith("age_") and c != "age_bracket"
            and c not in ("age_x_is_rb", "age_x_is_qb")  # Already included above
        ]

        X = train_df[feature_cols].fillna(0)
        y = train_df["target_ktc"]

        return X, y

    def get_weekly_feature_matrix(self) -> tuple[pd.DataFrame, pd.Series]:
        """Prepare features and target for weekly KTC change prediction.

        Creates week-over-week training pairs where:
        - Features come from week N
        - Target is ktc_change (week N+1 KTC - week N KTC)

        Returns ~60,000 weekly pairs for training.
        """
        players = self.get_players()
        training_rows = []

        for player in players:
            position = player["position"]

            for season in player.get("seasons", []):
                weekly_stats = season.get("weekly_stats", [])
                weekly_ktc = season.get("weekly_ktc", [])
                age = season.get("age", 25)
                years_exp = season.get("years_exp", 0)

                # Need matching weekly data
                if not weekly_stats or not weekly_ktc:
                    continue

                # Create week-over-week pairs
                for week_idx in range(len(weekly_stats) - 1):
                    # Get current and next week data
                    if week_idx >= len(weekly_ktc) or week_idx + 1 >= len(weekly_ktc):
                        continue

                    current_stats = weekly_stats[week_idx]
                    current_ktc_data = weekly_ktc[week_idx]
                    next_ktc_data = weekly_ktc[week_idx + 1]

                    current_ktc = current_ktc_data.get("ktc", 0)
                    next_ktc = next_ktc_data.get("ktc", 0)

                    # Skip invalid KTC values
                    if current_ktc <= 0 or next_ktc <= 0:
                        continue

                    # Calculate YTD features (up to and including current week)
                    ytd_fp = sum(
                        w.get("fantasy_points", 0)
                        for w in weekly_stats[: week_idx + 1]
                    )
                    ytd_games = sum(
                        w.get("games_played", 0)
                        for w in weekly_stats[: week_idx + 1]
                    )

                    # Target: KTC change
                    ktc_change = next_ktc - current_ktc

                    row = {
                        "current_ktc": current_ktc,
                        "weekly_fantasy_points": current_stats.get("fantasy_points", 0),
                        "games_played_this_week": current_stats.get("games_played", 0),
                        "week_of_season": current_stats.get("week", week_idx + 1),
                        "ytd_fantasy_points": ytd_fp,
                        "ytd_games": ytd_games,
                        "age": age,
                        "years_exp": years_exp,
                        "position": position,
                        # Target
                        "ktc_change": ktc_change,
                    }
                    training_rows.append(row)

        train_df = pd.DataFrame(training_rows)

        if train_df.empty:
            raise ValueError("No valid weekly training pairs found")

        # One-hot encode position
        position_dummies = pd.get_dummies(train_df["position"], prefix="pos")
        train_df = pd.concat([train_df, position_dummies], axis=1)

        # Feature columns
        feature_cols = [
            "current_ktc",
            "weekly_fantasy_points",
            "games_played_this_week",
            "week_of_season",
            "ytd_fantasy_points",
            "ytd_games",
            "age",
            "years_exp",
        ] + [c for c in train_df.columns if c.startswith("pos_")]

        X = train_df[feature_cols].fillna(0)
        y = train_df["ktc_change"]

        return X, y


# Singleton instance
_data_loader: Optional[DataLoader] = None


def get_data_loader() -> DataLoader:
    global _data_loader
    if _data_loader is None:
        _data_loader = DataLoader()
    return _data_loader
