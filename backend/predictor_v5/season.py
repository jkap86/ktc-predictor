"""Season tool: type a player's projected season points, get their predicted
end-of-season value from their pre-season value.

Start values are each player's value the day before kickoff, exported from
the thelab database with ``sql/start_values.sql`` into
``data/start_values_<season>.csv``. Every player gets two predictions, one
per label source, each starting from that source's own pre-season value:

- ``backchannel``: model trained on Backchannel trade-derived values
  (seasons 2022-2025, rows with >= 5 trades at both ends). The primary one.
- ``ktc``: model trained on KTC values (2020-2025), shown beside it.

The two scales differ, so compare each prediction with its own start value.
Models are trained from the committed data on first use and cached in
``artifacts/``.
"""

import math
from dataclasses import dataclass
from datetime import date
from functools import cached_property
from pathlib import Path

import pandas as pd

from predictor_v5.data import POSITIONS, build_dataset, load_season_stats
from predictor_v5.labels import CsvLabels, KtcLabels
from predictor_v5.model import WhatIfModel

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


def train_source_model(name: str, seasons: list[int] | None = None) -> WhatIfModel:
    source = label_source(name)
    df = build_dataset(load_season_stats(), source)
    df = df[df["season"].isin(seasons or SOURCES[name]["train_seasons"])]
    return WhatIfModel(label_source=source.name, scale_max=source.scale_max).fit(df)


def load_or_train(name: str, artifacts: Path = ARTIFACTS) -> WhatIfModel:
    """The saved season-tool model for a source, trained and saved if missing."""
    path = artifacts / f"season_{name}.joblib"
    if path.exists():
        return WhatIfModel.load(path)
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
        df["age"] = [age_on(b, self.cutoff) for b in df["birth_date"]]
        return df.reset_index(drop=True)

    @cached_property
    def models(self) -> dict[str, WhatIfModel]:
        return {name: load_or_train(name, self.artifacts) for name in SOURCES}

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

    def predict(self, player_id: str, points: list[float], age: float | None = None) -> dict:
        """Predicted end-of-season value per source for each points total."""
        info = self.player(player_id)
        age = info["age"] if age is None else age
        for name in SOURCES:
            src = info[name]
            start = src["start_value"]
            if not start or age is None:
                src["predictions"] = None
                src["note"] = "No age on file" if age is None else f"No pre-season {SOURCES[name]['label']} value"
                continue
            rows = pd.DataFrame({"start_value": start, "age": age, "position": info["position"], "points": points})
            values = self.models[name].predict(rows)
            src["predictions"] = [
                {"points": float(p), "value": round(float(v)), "change": round(float(v) - start)}
                for p, v in zip(points, values)
            ]
            src["train_seasons"] = self.models[name].train_seasons
            src["note"] = None
            if name == "backchannel" and (src.get("trades") or 0) < MIN_TRADES:
                src["note"] = (
                    f"Only {int(src.get('trades') or 0)} trades before the cutoff, so the start value is mostly "
                    "Backchannel's ADP prior. The model trained on players with 5 or more."
                )
        info["age"] = age
        info["season"] = self.season
        info["cutoff"] = self.cutoff.isoformat()
        return info
