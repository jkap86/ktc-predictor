"""Player-season stats and the joined training table.

Stats (position, age, season points) come from ``training-data.zip``
whichever label source is used, so switching labels never changes the
points definition. Points are full-PPR season totals as stored by the
existing data builder (4 pt passing TD).
"""

import json
import zipfile
from pathlib import Path

import pandas as pd

from predictor_v5.labels import LabelSource

POSITIONS = ("QB", "RB", "TE", "WR")

DEFAULT_ZIP = Path(__file__).resolve().parents[1] / "data" / "training-data.zip"


def read_training_players(zip_path: Path | str = DEFAULT_ZIP) -> list[dict]:
    with zipfile.ZipFile(zip_path) as zf:
        name = next(n for n in zf.namelist() if n.endswith(".json"))
        return json.loads(zf.read(name))["players"]


def load_season_stats(zip_path: Path | str = DEFAULT_ZIP) -> pd.DataFrame:
    """One row per (player_id, season) with position, age and points."""
    return drop_pre_nfl(read_season_stats(zip_path))


def read_season_stats(zip_path: Path | str = DEFAULT_ZIP) -> pd.DataFrame:
    """Like ``load_season_stats`` but keeping pre-NFL seasons, for coverage counts."""
    rows = []
    for p in read_training_players(zip_path):
        if p["position"] not in POSITIONS:
            continue
        for s in p.get("seasons", []):
            rows.append(
                {
                    "player_id": str(p["player_id"]),
                    "name": p.get("name"),
                    "season": int(s["year"]),
                    "position": p["position"],
                    "age": float(s["age"]),
                    "points": float(s.get("fantasy_points") or 0.0),
                    "games": int(s.get("games_played") or 0),
                    "years_exp": int(s["years_exp"]) if s.get("years_exp") is not None else -1,
                }
            )
    return pd.DataFrame(rows)


def drop_pre_nfl(stats: pd.DataFrame) -> pd.DataFrame:
    """Drop college (devy) seasons.

    KTC values players before they're drafted, so the zip has 0-point
    seasons such as Caleb Williams 2023. In the zip a player's rookie NFL
    season has ``years_exp`` 1, so ``years_exp == 0`` with no games is the
    college season before the draft (169 labelled rows). ``rookie_year``
    doesn't flag these reliably. Rookies and veterans who didn't play, such
    as Jordan Love 2020, stay.
    """
    college = (stats["years_exp"] == 0) & (stats["games"] == 0)
    return stats[~college].reset_index(drop=True)


def has_positive_values(df: pd.DataFrame) -> pd.Series:
    """Rows the model can train on: log(end / start) needs both values > 0."""
    return (df["start_value"] > 0) & (df["end_value"] > 0)


def build_dataset(stats: pd.DataFrame, labels: LabelSource) -> pd.DataFrame:
    """Join stats to start/end values; keep rows where both values are > 0."""
    values = labels.load()
    df = stats.merge(values, on=["player_id", "season"], how="inner")
    df = df[has_positive_values(df)]
    return df.sort_values(["season", "player_id"]).reset_index(drop=True)
