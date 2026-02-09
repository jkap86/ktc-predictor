"""Build training data for week-to-week KTC transitions.

Each row represents a (week_t → week_{t+1}) transition with features
computed from cumulative stats through week_t.
"""

import json
import zipfile
from pathlib import Path

import numpy as np
import pandas as pd

VALID_POSITIONS = {"QB", "RB", "WR", "TE"}

# Position-specific prime ages for age_prime_distance feature
PRIME_AGES = {"QB": 27, "RB": 24, "WR": 26, "TE": 27}


def build_transition_df(
    zip_path: str = "data/training-data.zip",
    json_name: str = "training-data.json",
) -> pd.DataFrame:
    """Build training data for week-to-week KTC transitions.

    Each row is a transition from week_t to week_{t+1} with:
    - ktc_current: KTC at week t
    - ktc_next: KTC at week t+1 (target)
    - ktc_delta_log: log(ktc_next / ktc_current) (target for training)
    - ppg_cumulative: PPG through week t
    - games_played: Games played through week t
    - weekly_fp: Fantasy points scored in week t
    - week: Current week number (1-18)
    - age, position, player_id, year

    Parameters
    ----------
    zip_path : str
        Path to the training data zip file.
    json_name : str
        Name of the JSON file inside the zip.

    Returns
    -------
    pd.DataFrame
        DataFrame with transition rows ready for training.
    """
    zip_path = Path(zip_path)
    with zipfile.ZipFile(zip_path, "r") as zf:
        with zf.open(json_name) as f:
            data = json.load(f)

    rows: list[dict] = []

    for player in data["players"]:
        pid = player["player_id"]
        position = player["position"]

        if position not in VALID_POSITIONS:
            continue

        for season in player.get("seasons", []):
            year = season["year"]
            age = season.get("age")
            weekly_ktc = season.get("weekly_ktc", [])
            weekly_stats = season.get("weekly_stats", [])

            if not weekly_ktc or not weekly_stats:
                continue

            # Build lookup for weekly stats
            stats_by_week = {ws["week"]: ws for ws in weekly_stats}

            # Build lookup for weekly KTC (filter valid values)
            ktc_by_week = {}
            for wk in weekly_ktc:
                week = wk["week"]
                ktc = wk.get("ktc", 0)
                if ktc and ktc > 0 and ktc < 9999:
                    ktc_by_week[week] = ktc

            # Sort weeks to find consecutive pairs
            sorted_weeks = sorted(ktc_by_week.keys())

            # Track cumulative stats
            cumulative_fp = 0.0
            cumulative_games = 0

            for i in range(len(sorted_weeks) - 1):
                week_t = sorted_weeks[i]
                week_t1 = sorted_weeks[i + 1]

                # Only use consecutive weeks
                if week_t1 != week_t + 1:
                    # Reset cumulative tracking if gap
                    # Actually, we should still accumulate stats even through gaps
                    pass

                ktc_current = ktc_by_week[week_t]
                ktc_next = ktc_by_week[week_t1]

                # Accumulate stats through week_t
                for w in range(1, week_t + 1):
                    if w in stats_by_week and w not in [
                        sw for sw in sorted_weeks[:i]
                    ]:
                        # Already counted in previous iterations
                        pass

                # More robust: compute cumulative from scratch each time
                cumulative_fp = 0.0
                cumulative_games = 0
                for w in range(1, week_t + 1):
                    if w in stats_by_week:
                        ws = stats_by_week[w]
                        cumulative_fp += ws.get("fantasy_points", 0) or 0
                        cumulative_games += ws.get("games_played", 0) or 0

                ppg_cumulative = (
                    cumulative_fp / cumulative_games if cumulative_games > 0 else 0.0
                )

                # This week's stats
                week_stats = stats_by_week.get(week_t, {})
                weekly_fp = week_stats.get("fantasy_points", 0) or 0
                games_this_week = week_stats.get("games_played", 0) or 0

                # Previous week's KTC for momentum (if available)
                ktc_prev = None
                if i > 0:
                    week_prev = sorted_weeks[i - 1]
                    if week_prev == week_t - 1:
                        ktc_prev = ktc_by_week.get(week_prev)

                # Compute features
                ktc_delta_log = np.log(ktc_next / ktc_current)
                ktc_current_log = np.log(ktc_current)
                season_progress = week_t / 18.0

                # KTC momentum: log(ktc_t / ktc_{t-1})
                ktc_momentum = None
                if ktc_prev is not None and ktc_prev > 0:
                    ktc_momentum = np.log(ktc_current / ktc_prev)

                # Age prime distance
                prime_age = PRIME_AGES.get(position, 26)
                age_prime_distance = (age - prime_age) if age else None

                rows.append(
                    {
                        "player_id": pid,
                        "position": position,
                        "year": year,
                        "week": week_t,
                        # Target
                        "ktc_next": ktc_next,
                        "ktc_delta_log": ktc_delta_log,
                        # Features
                        "ktc_current": ktc_current,
                        "ktc_current_log": ktc_current_log,
                        "ppg_cumulative": ppg_cumulative,
                        "games_played": cumulative_games,
                        "weekly_fp": weekly_fp,
                        "games_this_week": games_this_week,
                        "season_progress": season_progress,
                        "ktc_momentum": ktc_momentum,
                        "age": age,
                        "age_prime_distance": age_prime_distance,
                    }
                )

    df = pd.DataFrame(rows)

    # Summary stats
    print(f"Built {len(df):,} transition rows")
    for pos in VALID_POSITIONS:
        pos_count = len(df[df["position"] == pos])
        print(f"  {pos}: {pos_count:,}")

    # Target distribution
    print(f"\nTarget (ktc_delta_log) stats:")
    print(f"  mean: {df['ktc_delta_log'].mean():.4f}")
    print(f"  std:  {df['ktc_delta_log'].std():.4f}")
    print(f"  min:  {df['ktc_delta_log'].min():.4f}")
    print(f"  max:  {df['ktc_delta_log'].max():.4f}")

    return df


# Feature definitions for training
TRANSITION_FEATURES = [
    "ktc_current_log",
    "ppg_cumulative",
    "games_played",
    "weekly_fp",
    "games_this_week",
    "season_progress",
    "ktc_momentum",
    "age_prime_distance",
]


def get_transition_features() -> list[str]:
    """Get the list of features used for transition model."""
    return TRANSITION_FEATURES.copy()


if __name__ == "__main__":
    # Quick test
    df = build_transition_df()
    print(f"\nDataFrame shape: {df.shape}")
    print(f"\nSample rows:")
    print(df.head(10).to_string())
