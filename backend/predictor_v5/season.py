"""Season tool: type a player's projected season points (and games), get
their predicted end-of-season value from their pre-season value.

Start values are each player's value the day before kickoff, exported from
the thelab database with ``sql/start_values.sql`` into
``data/start_values_<season>.csv`` (2026: Backchannel model
``baseline-hl28-v1``, 528 players). Backchannel's scale shrinks every season
(stars started near 8,000 in 2022, 5,000 in 2025 and 2,900 in 2026), so read
its predictions as a change from the player's own start value. Every player gets two predictions, one
per label source, each starting from that source's own pre-season value:

- ``backchannel``: model trained on Backchannel trade-derived values
  (seasons 2022-2025, rows with >= 5 trades at both ends). The primary one.
- ``ktc``: model trained on KTC values (2020-2025), shown beside it.

The two scales differ, so compare each prediction with its own start value.
Models are the v2 what-if model (``whatif_v2.py``: points, games played,
points per game and prior-year points). Prior-year points come from the
stats archive; games default to a full 17-game season. Models are trained
from the committed data on first use and cached in ``artifacts/``.
"""

import math
from dataclasses import dataclass
from datetime import date
from functools import cached_property
from pathlib import Path

import pandas as pd

from predictor_v5.data import POSITIONS
from predictor_v5.labels import CsvLabels, KtcLabels
from predictor_v5.whatif_v2 import FULL_SEASON_GAMES, WhatIfModelV2, build_dataset_v2, prior_points_for

PACKAGE = Path(__file__).resolve().parent
DATA = PACKAGE / "data"
ARTIFACTS = PACKAGE / "artifacts"

BACKCHANNEL_LABELS = DATA / "backchannel_labels_2019_2025.csv"
# Training used rows with at least this many trades in the start window;
# a start value fitted from fewer is mostly Backchannel's ADP prior.
MIN_TRADES = 5

# The day before the first game of each season. Start values are read as of
# this date.
CUTOFFS = {2026: date(2026, 9, 9)}

SOURCES = {
    "backchannel": {
        "label": "Backchannel",
        "train_seasons": [2022, 2023, 2024, 2025],
        "value_column": "bc_value",
        "date_column": "bc_date",
    },
    "ktc": {
        "label": "KTC",
        "train_seasons": [2020, 2021, 2022, 2023, 2024, 2025],
        "value_column": "ktc_value",
        "date_column": "ktc_date",
    },
}


def label_source(name: str):
    if name == "backchannel":
        return CsvLabels(BACKCHANNEL_LABELS, name="backchannel", min_trades=MIN_TRADES)
    if name == "ktc":
        return KtcLabels()
    raise ValueError(f"Unknown source {name!r}")


def train_source_model(name: str, seasons: list[int] | None = None) -> WhatIfModelV2:
    source = label_source(name)
    df = build_dataset_v2(source)
    df = df[df["season"].isin(seasons or SOURCES[name]["train_seasons"])]
    return WhatIfModelV2(label_source=source.name, scale_max=source.scale_max).fit(df)


def load_or_train(name: str, artifacts: Path = ARTIFACTS) -> WhatIfModelV2:
    """The saved season-tool model for a source, trained and saved if missing."""
    path = artifacts / f"season_{name}_v2.joblib"
    if path.exists():
        return WhatIfModelV2.load(path)
    model = train_source_model(name)
    model.save(path)
    return model


def age_on(birth_date: str | None, on: date) -> float | None:
    """Whole years on ``on``, matching the integer ages the model trained on."""
    if not isinstance(birth_date, str) or not birth_date:
        return None
    born = date.fromisoformat(birth_date[:10])
    return float(on.year - born.year - ((on.month, on.day) < (born.month, born.day)))


def _num(x) -> float | None:
    return None if x is None or (isinstance(x, float) and math.isnan(x)) else float(x)


@dataclass
class SeasonTool:
    season: int = 2026
    start_values_path: Path | None = None
    artifacts: Path = ARTIFACTS

    @property
    def cutoff(self) -> date:
        return CUTOFFS[self.season]

    @cached_property
    def players(self) -> pd.DataFrame:
        path = self.start_values_path or DATA / f"start_values_{self.season}.csv"
        if not Path(path).exists():
            raise FileNotFoundError(
                f"{path} not found. Export it from the thelab database with predictor_v5/sql/start_values.sql "
                f"(season={self.season}, cutoff={self.cutoff})."
            )
        df = pd.read_csv(path, dtype={"player_id": str})
        df = df[df["position"].isin(POSITIONS)].copy()
        # KTC first: Backchannel's untraded players carry prior values that
        # don't rank them meaningfully.
        df = df.sort_values(["ktc_value", "bc_value"], ascending=False, na_position="last")
        df["age"] = [age_on(b, self.cutoff) for b in df["birth_date"]]
        return df.reset_index(drop=True)

    @cached_property
    def models(self) -> dict[str, WhatIfModelV2]:
        return {name: load_or_train(name, self.artifacts) for name in SOURCES}

    @cached_property
    def prior_points(self) -> dict[str, float]:
        """Each player's points in the season before this one, from the stats archive."""
        return prior_points_for(self.season)

    def search(self, query: str = "", position: str | None = None, limit: int = 25) -> list[dict]:
        df = self.players
        if query:
            df = df[df["name"].str.contains(query, case=False, na=False, regex=False)]
        if position:
            df = df[df["position"] == position.upper()]
        return [self._player(row) for _, row in df.head(limit).iterrows()]

    def find(self, query: str) -> list[dict]:
        """Exact player id, else exact name, else name substring matches."""
        df = self.players
        for mask in (df["player_id"] == query, df["name"].str.lower() == query.lower()):
            if mask.any():
                return [self._player(row) for _, row in df[mask].iterrows()]
        return self.search(query, limit=10)

    def player(self, player_id: str) -> dict:
        hit = self.players[self.players["player_id"] == str(player_id)]
        if hit.empty:
            raise KeyError(f"No {self.season} start value for player {player_id}")
        return self._player(hit.iloc[0])

    def _player(self, row: pd.Series) -> dict:
        out = {
            "player_id": row["player_id"],
            "name": row["name"],
            "position": row["position"],
            "team": row["team"] if isinstance(row.get("team"), str) else None,
            "age": _num(row["age"]),
        }
        for name, cfg in SOURCES.items():
            out[name] = {
                "start_value": _num(row.get(cfg["value_column"])),
                "start_date": row.get(cfg["date_column"]) if isinstance(row.get(cfg["date_column"]), str) else None,
            }
        out["backchannel"]["trades"] = _num(row.get("bc_trades"))
        return out

    def predict(
        self,
        player_id: str,
        points: list[float],
        age: float | None = None,
        games: float | None = None,
    ) -> dict:
        """Predicted end-of-season value per source for each points total.

        ``games`` is the projected games played behind those points (a full
        season when omitted); prior-year points come from the stats archive.
        """
        info = self.player(player_id)
        age = info["age"] if age is None else age
        games = FULL_SEASON_GAMES if games is None else float(games)
        if games < 0:
            raise ValueError("games must be >= 0")
        prior_points = self.prior_points.get(str(player_id), 0.0)
        for name in SOURCES:
            src = info[name]
            start = src["start_value"]
            if not start or age is None:
                src["predictions"] = None
                src["note"] = "No age on file" if age is None else f"No pre-season {SOURCES[name]['label']} value"
                continue
            trades = int(src.get("trades") or 0)
            if name == "backchannel" and trades < MIN_TRADES:
                # Untraded players sit at Backchannel's ADP prior, which puts
                # some long-gone players near the top of the board.
                src["predictions"] = None
                src["note"] = (
                    f"Only {trades} trades before the cutoff, so there's no trade-based start value. "
                    f"The model needs at least {MIN_TRADES}."
                )
                continue
            rows = pd.DataFrame(
                {
                    "start_value": start,
                    "age": age,
                    "position": info["position"],
                    "points": points,
                    "games": games,
                    "prior_points": prior_points,
                }
            )
            values = self.models[name].predict(rows)
            src["predictions"] = [
                {"points": float(p), "value": round(float(v)), "change": round(float(v) - start)}
                for p, v in zip(points, values)
            ]
            src["train_seasons"] = self.models[name].train_seasons
            src["note"] = None
        info["age"] = age
        info["games"] = games
        info["prior_points"] = prior_points
        info["season"] = self.season
        info["cutoff"] = self.cutoff.isoformat()
        return info
