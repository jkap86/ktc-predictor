"""Short-window tool: what 3 or 6 games do to a player's dynasty value.

Pick a player, a window of 3 or 6 games and the week it starts. The tool
takes the player's value on the Tuesday before that week, their form so far
this season and the points they score in the window, and predicts their
value on the Tuesday after it. For a finished window the points default to
what the player actually scored, and the tool also shows the actual value
now and what a window at the player's own season pace would have predicted,
so the difference is what those games did. For a window still to come, give
the points as a what-if (``points``, or several for a sweep).

Models are ``short_window.py``'s full feature set, one per source and window
length, trained on every season (KTC 2021-25; Backchannel 2020-25, windows
with 5+ trades at both ends), each with an 80% band from held-out residuals
(trained without the last season, scored on it), as v2 does.

- **KTC** is the main source.
- **Backchannel** is shown only for players with 50+ trades in the 150-day
  window at the window's start: below that, short moves are partly noise
  (README, "Short windows"). Local ``baseline-hl28-v1`` history stops on
  2026-10-09, so 2026 windows starting after week 5 have no Backchannel
  value.

Late season (window starting week 14 or later) gets a note: on KTC the
points there move value much less (skill 0.08 against 0.23 for 3 games).

Live inputs come from ``sql/window_inputs.sql`` into
``data/window_inputs_<season>.csv``.

    cd backend
    python -m predictor_v5.window_tool --player "Ja'Marr Chase" --games 3              # latest finished 3 games
    python -m predictor_v5.window_tool --player "Ja'Marr Chase" --games 3 --start-week 9
    python -m predictor_v5.window_tool --player "Ja'Marr Chase" --games 6 --start-week 12 --sweep 60,90,120
"""

import argparse
import json
import math
from dataclasses import dataclass, field
from datetime import date
from functools import cached_property
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

from predictor_v5.data import POSITIONS
from predictor_v5.short_window import (
    DEFAULT_VALUES,
    FEATURES,
    build_windows,
    fit,
    read_bc_values,
    read_zip,
    window_inputs,
)
from predictor_v5.whatif_v2 import prior_points_for

PACKAGE = Path(__file__).resolve().parent
DATA = PACKAGE / "data"
ARTIFACTS = PACKAGE / "artifacts"

GAMES = (3, 6)
WEEKS = 18
LATE_WEEK = 14
PACE_PRIOR_GAMES = 3
BAND = (0.1, 0.9)
FEATURE_SET = FEATURES["full"]
WEEK1 = {2026: date(2026, 9, 8)}
# The day start values were read (the season tool's cutoff); ages are whole years then.
AGE_ON = {2026: date(2026, 9, 9)}

SOURCES = {
    "ktc": {"label": "KTC", "train_from": 2021, "min_trades": 0, "last_value_week": 18, "min_start_trades": 0,
            "value_column": "ktc_value", "date_column": "ktc_date"},
    "backchannel": {"label": "Backchannel", "train_from": 2020, "min_trades": 5, "last_value_week": None,
                    "min_start_trades": 50, "value_column": "bc_value", "date_column": "tuesday"},
}
# Late-season skill from the 2022-25 folds (README), quoted in the note.
LATE_NOTE = {
    3: "Late-season windows move KTC value much less: points explain about 8% of the error then, against 23% earlier.",
    6: "Late-season windows move value less than earlier ones.",
}


def _num(x) -> float | None:
    return None if x is None or (isinstance(x, float) and math.isnan(x)) else float(x)


@dataclass
class WindowModel:
    source: str
    games: int
    estimator: object = None
    train_seasons: list[int] = field(default_factory=list)
    band_offsets: dict[float, float] = field(default_factory=dict)
    band_season: int | None = None
    train_rows: int = 0

    def predict_points(self, frame, points: list[float]) -> dict[str, np.ndarray]:
        """Predictions for one player at each window points total, made monotone.

        ``frame(list_of_points)`` builds the feature rows. Like v2, the curve
        is a running max over a 1-point grid from 0, and each total reads the
        grid point at or below it, so more points never predict less (the
        trees' monotone constraints leave tiny dips after the leaf refit).
        """
        grid = np.arange(int(np.floor(max(points))) + 1, dtype=float)
        curve = np.maximum.accumulate(self.estimator.predict(frame(list(grid))[FEATURE_SET].to_numpy(float)))
        idx = np.floor(np.asarray(points, float)).astype(int)
        return self.predict(frame(list(points)), curve[idx])

    def fit(self, df: pd.DataFrame) -> "WindowModel":
        self.estimator = fit(df, FEATURE_SET)
        self.train_seasons = sorted(int(s) for s in df["season"].unique())
        self.train_rows = int(len(df))
        last = max(self.train_seasons)
        inner = fit(df[df["season"] < last], FEATURE_SET)
        held = df[df["season"] == last]
        resid = held["log_change"].to_numpy() - inner.predict(held[FEATURE_SET].to_numpy(float))
        self.band_offsets = {q: float(np.quantile(resid, q)) for q in BAND}
        self.band_season = last
        return self

    def predict(self, df: pd.DataFrame, log_ratio: np.ndarray | None = None) -> dict[str, np.ndarray]:
        if log_ratio is None:
            log_ratio = self.estimator.predict(df[FEATURE_SET].to_numpy(float))
        start = df["start_value"].to_numpy(float)
        out = {"value": start * np.exp(log_ratio)}
        out["low"] = start * np.exp(log_ratio + self.band_offsets[BAND[0]])
        out["high"] = start * np.exp(log_ratio + self.band_offsets[BAND[1]])
        return out


def train(source: str, games: int, values_path: Path = DEFAULT_VALUES) -> WindowModel:
    cfg = SOURCES[source]
    pts, ktc = read_zip()
    values = ktc if source == "ktc" else read_bc_values(values_path)
    df = build_windows(pts, values, games, cfg["min_trades"], cfg["last_value_week"])
    df = df[df["season"] >= cfg["train_from"]]
    return WindowModel(source=source, games=games).fit(df)


def load_or_train(source: str, games: int, artifacts: Path = ARTIFACTS) -> WindowModel:
    path = artifacts / f"window_{source}_{games}.joblib"
    if path.exists():
        return joblib.load(path)
    model = train(source, games)
    path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(model, path)
    return model


def age_on(birth_date, on: date) -> float | None:
    if not isinstance(birth_date, str) or not birth_date:
        return None
    born = date.fromisoformat(birth_date[:10])
    return float(on.year - born.year - ((on.month, on.day) < (born.month, born.day)))


@dataclass
class WindowTool:
    season: int = 2026
    inputs_path: Path | None = None
    artifacts: Path = ARTIFACTS

    @cached_property
    def inputs(self) -> pd.DataFrame:
        path = self.inputs_path or DATA / f"window_inputs_{self.season}.csv"
        if not Path(path).exists():
            raise FileNotFoundError(f"{path} not found. Export it with predictor_v5/sql/window_inputs.sql.")
        df = pd.read_csv(path, dtype={"player_id": str})
        bc_path = DATA / f"bc_weekly_{self.season}.csv" if self.inputs_path is None else None
        if bc_path is not None and bc_path.exists():
            # Backchannel values exported from the full-corpus database
            # (sql/bc_weekly.sql) replace the local ones outright, so every
            # week comes from one trade corpus; weeks it lacks have none.
            bc = pd.read_csv(bc_path, dtype={"player_id": str})[["player_id", "week", "bc_value", "bc_trades"]]
            df = df.drop(columns=["bc_value", "bc_trades"]).merge(bc, on=["player_id", "week"], how="left")
        return df[df["position"].isin(POSITIONS)]

    @cached_property
    def players(self) -> pd.DataFrame:
        df = self.inputs
        latest = df.dropna(subset=["ktc_value"]).sort_values("week").groupby("player_id").tail(1)
        info = df.drop_duplicates("player_id")[["player_id", "name", "position", "team", "birth_date"]]
        out = info.merge(latest[["player_id", "ktc_value"]], on="player_id", how="left")
        return out.sort_values("ktc_value", ascending=False, na_position="last").reset_index(drop=True)

    @cached_property
    def last_played_week(self) -> int:
        """The latest week with any points on file: windows up to it are finished."""
        played = self.inputs.dropna(subset=["points"])
        return int(played["week"].max()) if len(played) else 0

    @cached_property
    def prior_points(self) -> dict[str, float]:
        return prior_points_for(self.season)

    def model(self, source: str, games: int) -> WindowModel:
        key = (source, games)
        if key not in self._models:
            self._models[key] = load_or_train(source, games, self.artifacts)
        return self._models[key]

    @cached_property
    def _models(self) -> dict:
        return {}

    def info(self) -> dict:
        return {"season": self.season, "games": list(GAMES), "last_played_week": self.last_played_week,
                "sources": {k: v["label"] for k, v in SOURCES.items()},
                "backchannel_min_trades": SOURCES["backchannel"]["min_start_trades"]}

    def search(self, query: str = "", position: str | None = None, limit: int = 25) -> list[dict]:
        df = self.players
        if query:
            df = df[df["name"].str.contains(query, case=False, na=False, regex=False)]
        if position:
            df = df[df["position"] == position.upper()]
        return [self._player(r) for _, r in df.head(limit).iterrows()]

    def find(self, query: str) -> list[dict]:
        df = self.players
        for mask in (df["player_id"] == query, df["name"].str.lower() == query.lower()):
            if mask.any():
                return [self._player(r) for _, r in df[mask].iterrows()]
        return self.search(query, limit=10)

    @staticmethod
    def _player(r: pd.Series) -> dict:
        return {"player_id": r["player_id"], "name": r["name"], "position": r["position"],
                "team": r["team"] if isinstance(r["team"], str) else None, "ktc_value": _num(r["ktc_value"])}

    def default_start_week(self, games: int) -> int:
        """The latest finished window of this length (at least week 1)."""
        return max(1, self.last_played_week - games + 1)

    def predict(self, player_id: str, games: int = 3, start_week: int | None = None,
                points: list[float] | None = None, window_games: float | None = None) -> dict:
        if games not in GAMES:
            raise ValueError(f"games must be one of {GAMES}")
        rows = self.inputs[self.inputs["player_id"] == str(player_id)].sort_values("week")
        if rows.empty:
            raise KeyError(f"No {self.season} inputs for player {player_id}")
        start_week = self.default_start_week(games) if start_week is None else int(start_week)
        if not 1 <= start_week <= WEEKS - games + 1:
            raise ValueError(f"start_week must be 1-{WEEKS - games + 1} for {games} games")
        end_week = start_week + games  # the Tuesday after the window
        last_game = end_week - 1
        if start_week - 1 > self.last_played_week:
            raise ValueError(f"Weeks before {start_week} haven't been played yet (last played: {self.last_played_week})")
        finished = last_game <= self.last_played_week
        by_week = rows.set_index("week")
        weeks = range(1, WEEKS + 1)
        P = by_week["points"].reindex(weeks).fillna(0.0).to_numpy(float)
        G = by_week["played"].reindex(weeks).fillna(0.0).to_numpy(float)
        actual_points = float(P[start_week - 1:last_game].sum()) if finished else None
        actual_games = float(G[start_week - 1:last_game].sum()) if finished else None
        if window_games is None:
            window_games = actual_games if finished else float(games)
        if not 0 <= window_games <= games:
            raise ValueError(f"window_games must be 0-{games}")
        if points is None:
            if not finished:
                raise ValueError(f"Weeks {start_week}-{last_game} aren't finished; give the points as a what-if")
            points = [actual_points]
        if window_games == 0 and any(p > 0 for p in points):
            raise ValueError("points need at least one game")

        head = rows.iloc[0]
        age0 = age_on(head["birth_date"], AGE_ON.get(self.season, WEEK1[self.season]))
        prior = self.prior_points.get(str(player_id), 0.0)
        out = {
            "player_id": str(player_id), "name": head["name"], "position": head["position"],
            "team": head["team"] if isinstance(head["team"], str) else None,
            "season": self.season, "games": games, "start_week": start_week, "last_game_week": last_game,
            "finished": finished, "window_games": window_games, "actual_points": None if actual_points is None else round(actual_points, 1),
            "late_season": start_week >= LATE_WEEK,
            "note": LATE_NOTE[games] if start_week >= LATE_WEEK else None,
        }
        for name, cfg in SOURCES.items():
            src = {"label": cfg["label"]}
            out[name] = src
            val = {int(w): _num(v) for w, v in by_week[cfg["value_column"]].items()}
            start = val.get(start_week)
            src["start_value"] = start
            src["start_date"] = by_week[cfg["date_column"]].get(start_week) if start else None
            end = val.get(end_week)
            src["actual_value"] = end
            src["actual_change"] = None if not (start and end) else round(end - start)
            if name == "backchannel":
                trades = _num(by_week["bc_trades"].get(start_week))
                src["trades"] = trades
                if not start:
                    src["predictions"], src["note"] = None, f"No Backchannel value for week {start_week} on file"
                    continue
                if (trades or 0) < cfg["min_start_trades"]:
                    src["predictions"] = None
                    src["note"] = (f"Only {int(trades or 0)} trades behind the week-{start_week} value; short moves need "
                                   f"{cfg['min_start_trades']}+ to stand out from noise")
                    continue
            if not start or age0 is None:
                src["predictions"], src["note"] = None, ("No age on file" if age0 is None else f"No {cfg['label']} value for week {start_week}")
                continue
            base = {"start_value": start, "log_start": math.log(start), "age": age0 + (start_week - 1) / 52.0,
                    "position_code": float(POSITIONS.index(head["position"])), "week": start_week,
                    "prior_points": prior}
            form = window_inputs(P, G, val, start_week, games)
            # The player's pace: points per game so far, shrunk toward last
            # season's with PACE_PRIOR_GAMES games' weight (one early game is noisy).
            prior_ppg = prior / 17.0
            std_points = form["std_ppg"] * form["std_games"]
            pace = (std_points + PACE_PRIOR_GAMES * prior_ppg) / (form["std_games"] + PACE_PRIOR_GAMES) if prior_ppg else form["std_ppg"]

            def frame(pts_list):
                return pd.DataFrame([{**base, **form, "win_points": p, "win_games": window_games,
                                      "win_ppg": p / window_games if window_games else 0.0} for p in pts_list])

            model = self.model(name, games)
            pred = model.predict_points(frame, points)
            src["predictions"] = [
                {"points": round(float(p), 1), "value": round(float(v)), "change": round(float(v) - start),
                 "low": round(float(lo)), "high": round(float(hi))}
                for p, v, lo, hi in zip(points, pred["value"], pred["low"], pred["high"])
            ]
            pace_pts = pace * window_games
            src["pace"] = {"ppg": round(pace, 2), "points": round(pace_pts, 1),
                           "value": round(float(model.predict_points(frame, [pace_pts])["value"][0]))}
            src["train_seasons"] = model.train_seasons
            src["band"] = {"low_quantile": BAND[0], "high_quantile": BAND[1], "calibration_season": model.band_season}
            src["note"] = None
        return out


def main(argv=None) -> None:
    ap = argparse.ArgumentParser(prog="python -m predictor_v5.window_tool")
    ap.add_argument("--player", required=True)
    ap.add_argument("--games", type=int, choices=GAMES, default=3)
    ap.add_argument("--start-week", type=int)
    ap.add_argument("--sweep", help="comma-separated window points (a what-if)")
    ap.add_argument("--window-games", type=float)
    ap.add_argument("--json", action="store_true")
    a = ap.parse_args(argv)
    tool = WindowTool()
    hits = tool.find(a.player)
    if len(hits) != 1:
        raise SystemExit(f"{len(hits)} players match {a.player!r}: " + ", ".join(h["name"] for h in hits))
    points = [float(x) for x in a.sweep.split(",")] if a.sweep else None
    r = tool.predict(hits[0]["player_id"], a.games, a.start_week, points, a.window_games)
    if a.json:
        print(json.dumps(r, indent=2, default=str))
        return
    print(f"{r['name']} ({r['position']}), weeks {r['start_week']}-{r['last_game_week']}, {r['window_games']:g} games"
          + (f", scored {r['actual_points']:.1f}" if r["finished"] else ""))
    if r["note"]:
        print("  " + r["note"])
    for name in SOURCES:
        s = r[name]
        print(f"  {s['label']}: start {s['start_value']}" + (f" ({s['start_date']})" if s.get("start_date") else ""))
        if not s["predictions"]:
            print(f"    {s['note']}")
            continue
        for p in s["predictions"]:
            print(f"    {p['points']:.1f} pts -> {p['value']} ({p['change']:+d}), 80% {p['low']}-{p['high']}")
        print(f"    at season pace ({s['pace']['ppg']} ppg, {s['pace']['points']} pts) -> {s['pace']['value']}")
        if s["actual_value"]:
            print(f"    actual: {s['actual_value']:.0f} ({s['actual_change']:+d})")


if __name__ == "__main__":
    # Run through the package module so cached models pickle as
    # predictor_v5.window_tool.WindowModel, not __main__.WindowModel.
    from predictor_v5.window_tool import main as _main

    _main()
