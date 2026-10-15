"""Label sources: where start- and end-of-season values come from.

A label source returns one row per (player_id, season) with
``start_value`` and ``end_value``. Player ids are Sleeper ids in every
source, so the stats join is the same whichever one is used.

- ``KtcLabels``: KTC start/end values from ``training-data.zip`` (0-9999).
- ``CsvLabels``: any export with the same four columns. Backchannel goes
  here: export ``reference_value`` (the fixed-curve scale, not the
  published value that renormalizes daily) at each season's start and end
  cutoffs.
"""

from dataclasses import dataclass
from pathlib import Path
from typing import Protocol

import pandas as pd

LABEL_COLUMNS = ["player_id", "season", "start_value", "end_value"]


class LabelSource(Protocol):
    name: str
    scale_max: float

    def load(self) -> pd.DataFrame: ...


@dataclass
class KtcLabels:
    zip_path: Path | str | None = None
    name: str = "ktc"
    scale_max: float = 9999.0

    def load(self) -> pd.DataFrame:
        from predictor_v5.data import DEFAULT_ZIP, read_training_players

        rows = []
        for p in read_training_players(self.zip_path or DEFAULT_ZIP):
            for s in p.get("seasons", []):
                rows.append(
                    {
                        "player_id": str(p["player_id"]),
                        "season": int(s["year"]),
                        "start_value": float(s.get("start_ktc") or 0.0),
                        "end_value": float(s.get("end_ktc") or 0.0),
                    }
                )
        return pd.DataFrame(rows, columns=LABEL_COLUMNS)


@dataclass
class CsvLabels:
    path: Path | str
    name: str = "csv"
    scale_max: float = 10000.0

    def load(self) -> pd.DataFrame:
        df = pd.read_csv(self.path, dtype={"player_id": str})
        missing = set(LABEL_COLUMNS) - set(df.columns)
        if missing:
            raise ValueError(f"{self.path} is missing columns: {sorted(missing)}")
        df = df[LABEL_COLUMNS].copy()
        df["season"] = df["season"].astype(int)
        return df


def get_label_source(spec: str) -> LabelSource:
    """``ktc`` or ``csv:<path>`` (``backchannel:<path>`` is an alias)."""
    if spec == "ktc":
        return KtcLabels()
    kind, _, path = spec.partition(":")
    if kind in ("csv", "backchannel") and path:
        return CsvLabels(path=path, name=kind)
    raise ValueError(f"Unknown label source {spec!r}; use 'ktc' or 'csv:<path>'")
