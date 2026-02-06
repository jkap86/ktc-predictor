"""Build weekly cumulative snapshot DataFrame from training data."""

import json
import zipfile
from pathlib import Path

import numpy as np
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
                 ppg_so_far, start_ktc, age, weeks_missed_so_far,
                 draft_pick, years_remaining, end_ktc, abs_change
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

            start_ktc = season.get("start_ktc")
            if start_ktc is None or start_ktc <= 0:
                continue

            year = season["year"]
            age = season.get("age")
            draft_pick = season.get("draft_pick")
            years_remaining = season.get("years_remaining")
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
                weeks_missed_so_far = ws["week"] - games_so_far

                rows.append(
                    {
                        "player_id": pid,
                        "position": position,
                        "year": year,
                        "week": ws["week"],
                        "games_played_so_far": games_so_far,
                        "ppg_so_far": round(ppg_so_far, 4),
                        "start_ktc": start_ktc,
                        "age": age,
                        "weeks_missed_so_far": weeks_missed_so_far,
                        "draft_pick": draft_pick,
                        "years_remaining": years_remaining,
                        "end_ktc": end_ktc,
                        "abs_change": end_ktc - start_ktc,
                        "log_ratio": np.log(max(end_ktc, 1) / start_ktc),
                    }
                )

    df = pd.DataFrame(rows)

    # Mark and replace 9999 sentinels in both start_ktc and end_ktc
    df["start_ktc_was_sentinel"] = 0
    for pos in VALID_POSITIONS:
        mask = df["position"] == pos
        if not mask.any():
            continue
        # start_ktc sentinels → median (conservative; flag compensates)
        start_sent = mask & (df["start_ktc"] >= 9999)
        if start_sent.any():
            non_sent = mask & (df["start_ktc"] < 9999)
            df.loc[start_sent, "start_ktc_was_sentinel"] = 1
            df.loc[start_sent, "start_ktc"] = df.loc[non_sent, "start_ktc"].median()
        # end_ktc sentinels → p95 (closer to truth for target)
        end_sent = mask & (df["end_ktc"] >= 9999)
        if end_sent.any():
            non_sent_end = mask & (df["end_ktc"] < 9999)
            df.loc[end_sent, "end_ktc"] = df.loc[non_sent_end, "end_ktc"].quantile(0.95)
        # Recompute derived columns for entire position
        df.loc[mask, "log_ratio"] = np.log(
            np.maximum(df.loc[mask, "end_ktc"], 1) / df.loc[mask, "start_ktc"]
        )
        df.loc[mask, "abs_change"] = df.loc[mask, "end_ktc"] - df.loc[mask, "start_ktc"]

    return df
