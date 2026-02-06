"""Build weekly cumulative snapshot DataFrame from training data."""

import json
import zipfile
from pathlib import Path

import pandas as pd

VALID_POSITIONS = {"QB", "RB", "WR", "TE"}


def build_weekly_snapshot_df(
    zip_path: str, json_name: str = "training-data.json"
) -> pd.DataFrame:
    """Load training data and produce weekly cumulative snapshots.

    For each player-season, walks through weekly_stats in week order,
    accumulating games_played and fantasy_points to compute a running ppg.

    Parameters
    ----------
    zip_path : str
        Path to the zip file containing the training JSON.
    json_name : str
        Name of the JSON file inside the zip.

    Returns
    -------
    pd.DataFrame
        Columns: player_id, position, year, week, games_played_so_far,
                 ppg_so_far, end_ktc
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

        for season in player["seasons"]:
            end_ktc = season.get("end_ktc")
            if end_ktc is None:
                continue

            year = season["year"]
            weekly_stats = season.get("weekly_stats", [])

            # Sort by week to ensure correct accumulation
            weekly_stats = sorted(weekly_stats, key=lambda w: w["week"])

            games_so_far = 0
            fp_so_far = 0.0

            for ws in weekly_stats:
                games_so_far += ws.get("games_played", 0)
                fp_so_far += ws.get("fantasy_points", 0.0)

                if games_so_far <= 0:
                    continue

                ppg_so_far = fp_so_far / games_so_far

                rows.append(
                    {
                        "player_id": pid,
                        "position": position,
                        "year": year,
                        "week": ws["week"],
                        "games_played_so_far": games_so_far,
                        "ppg_so_far": round(ppg_so_far, 4),
                        "end_ktc": end_ktc,
                    }
                )

    df = pd.DataFrame(rows)
    return df
