"""Weekly-sequence what-if model: the season as 18 weekly point totals, not one number.

v2 sees a season as its points total, games and points per game. The one
feature that moved the needle in the 2026-11-21 ablation was *when* the
points came (`last_4_vs_season`, +1.3 points of skill), which says the market
reads the shape of a season, not just its sum. This model gives the same
gradient-boosted learner the sequence itself, on top of v2's features. The
default is four quarter sums (weeks 1-4, 5-8, 9-13, 14-18); ``--sequence``
also offers three shape summaries (late form, trend, boom weeks), both, or
the raw 18 weeks (0 where the player did not play). Each sequence total is
constrained monotone increasing, so more points in any week never lowers the
prediction, and the total stays monotone with it (up to the few hundredths of
a percent that ``absolute_error``'s leaf refit allows; v2 closes that with a
points grid, which has no analogue over 18 weeks).

This is a research ceiling, not a season-tool replacement: the tool's user
gives one projected total, not a weekly line. If the sequence helps, the
next step is a weekly profile sampled from the total.

``evaluate`` runs v2's chronological folds for both models. With
``--value-weeks k`` it adds the player's own value path through week k
(``log(value in week w / start)`` for w <= k) and scores a mid-season update
instead: a different task, reported apart.

``model.py``, ``data.py``, ``evaluate.py`` and ``__main__.py`` are bound by
the sealed 2026 protocols; this module imports from them and changes nothing.

    cd backend
    python -m predictor_v5.weekly_model evaluate                       # KTC folds
    python -m predictor_v5.weekly_model --labels csv:predictor_v5/data/backchannel_labels_2019_2025_mfl_full.csv \\
        --min-trades 5 --training-data data/training-data-2019.zip --name mfl_sf evaluate
"""

import argparse
import json
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingRegressor

from predictor_v5.data import (
    DEFAULT_ZIP,
    POSITIONS,
    build_dataset,
    drop_pre_nfl,
    read_training_players,
)
from predictor_v5.evaluate import _mae, season_folds, summarize
from predictor_v5.labels import LabelSource, get_label_source
from predictor_v5.model import HGB_PARAMS
from predictor_v5.whatif_v2 import (
    BAND_QUANTILES,
    WhatIfModelV2,
    band_summary,
    points_per_game,
    read_season_stats_v2,
)

ARTIFACTS = Path(__file__).resolve().parent / "artifacts"
WEEKS = 18
WEEK_COLUMNS = [f"wk_{w}" for w in range(1, WEEKS + 1)]
# Four stretches of the season: weeks 1-4, 5-8, 9-13, 14-18.
QUARTERS = ((0, 4), (4, 8), (8, 13), (13, 18))
QUARTER_COLUMNS = [f"q{i + 1}" for i in range(len(QUARTERS))]
SHAPE_COLUMNS = ["late_form", "trend", "boom_weeks"]
# How the sequence reaches the learner. 18 raw weeks overfit at ~1,000 training
# rows (2026-11-23: 2.5 points below v2 on both sources); four quarter sums hold
# the shape and gained 2.6 points on the trade values, nothing on KTC.
SEQUENCES = {
    "quarters": QUARTER_COLUMNS,
    "shape": SHAPE_COLUMNS,
    "quarters+shape": QUARTER_COLUMNS + SHAPE_COLUMNS,
    "weeks": WEEK_COLUMNS,
}
DEFAULT_SEQUENCE = "quarters"
BASE_FEATURES = ["log_start", "age", "points", "games", "ppg", "prior_points", "position_code"]
BOOM_POINTS = 20.0
LATE_WEEKS = 4


def weekly_matrix(seasons: list[dict]) -> np.ndarray:
    """Points by week for each season dict, weeks 1..WEEKS, 0 where absent."""
    out = np.zeros((len(seasons), WEEKS))
    for i, s in enumerate(seasons):
        for w in s.get("weekly_stats") or []:
            week = int(w.get("week") or 0)
            if 1 <= week <= WEEKS:
                out[i, week - 1] = float(w.get("fantasy_points") or 0.0)
    return out


def shape_features(weekly: np.ndarray, games: np.ndarray) -> dict[str, np.ndarray]:
    """Summaries of a season's shape from its weekly points.

    ``late_form``: mean of the last LATE_WEEKS weeks with points, minus the
    season's points per game (positive: finished stronger than the average).
    ``trend``: least-squares slope of points over the weeks with points, in
    points per week. ``boom_weeks``: weeks at or above BOOM_POINTS.
    """
    n = weekly.shape[0]
    late = np.zeros(n)
    trend = np.zeros(n)
    ppg = points_per_game(weekly.sum(axis=1), games)
    for i in range(n):
        played = np.flatnonzero(weekly[i] > 0)
        if len(played) == 0:
            continue
        late[i] = weekly[i, played[-LATE_WEEKS:]].mean() - ppg[i]
        if len(played) >= 3:
            trend[i] = np.polyfit(played.astype(float), weekly[i, played], 1)[0]
    return {"late_form": late, "trend": trend, "boom_weeks": (weekly >= BOOM_POINTS).sum(axis=1).astype(float)}


def quarter_sums(weekly: np.ndarray) -> np.ndarray:
    return np.column_stack([weekly[:, a:b].sum(axis=1) for a, b in QUARTERS])


def read_weekly_stats(zip_path: Path | str = DEFAULT_ZIP) -> pd.DataFrame:
    """v2's season stats plus the weekly points matrix, its quarter sums and its shape summaries."""
    stats = read_season_stats_v2(zip_path)
    keyed, seasons = [], []
    for p in read_training_players(zip_path):
        if p["position"] not in POSITIONS:
            continue
        for s in p.get("seasons", []):
            keyed.append((str(p["player_id"]), int(s["year"])))
            seasons.append(s)
    weekly = weekly_matrix(seasons)
    frame = pd.DataFrame(weekly, columns=WEEK_COLUMNS)
    frame["player_id"] = [k[0] for k in keyed]
    frame["season"] = [k[1] for k in keyed]
    merged = stats.merge(frame, on=["player_id", "season"], how="left")
    merged[WEEK_COLUMNS] = merged[WEEK_COLUMNS].fillna(0.0)
    weekly = merged[WEEK_COLUMNS].to_numpy(float)
    merged[QUARTER_COLUMNS] = quarter_sums(weekly)
    shape = shape_features(weekly, merged["games"].to_numpy(float))
    for name, values in shape.items():
        merged[name] = values
    return merged


def load_weekly_stats(zip_path: Path | str = DEFAULT_ZIP) -> pd.DataFrame:
    return drop_pre_nfl(read_weekly_stats(zip_path))


def build_dataset_weekly(labels: LabelSource, zip_path: Path | str = DEFAULT_ZIP) -> pd.DataFrame:
    return build_dataset(load_weekly_stats(zip_path), labels)


def value_path_columns(k: int) -> list[str]:
    return [f"val_wk_{w}" for w in range(1, k + 1)]


def feature_names(value_weeks: int = 0, sequence: str = DEFAULT_SEQUENCE) -> list[str]:
    if sequence not in SEQUENCES:
        raise ValueError(f"sequence must be one of {sorted(SEQUENCES)}, not {sequence!r}")
    return BASE_FEATURES + SEQUENCES[sequence] + value_path_columns(value_weeks)


def make_features_weekly(df: pd.DataFrame, value_weeks: int = 0, sequence: str = DEFAULT_SEQUENCE) -> np.ndarray:
    unknown = set(df["position"]) - set(POSITIONS)
    if unknown:
        raise ValueError(f"Unknown positions: {sorted(unknown)}")
    points = df["points"].astype(float).to_numpy()
    games = df["games"].astype(float).to_numpy()
    columns = [
        np.log(df["start_value"].astype(float).to_numpy()),
        df["age"].astype(float).to_numpy(),
        points,
        games,
        points_per_game(points, games),
        df["prior_points"].astype(float).to_numpy(),
        df["position"].map(POSITIONS.index).astype(float).to_numpy(),
    ]
    columns += [df[c].astype(float).to_numpy() for c in SEQUENCES[sequence] + value_path_columns(value_weeks)]
    return np.column_stack(columns)


def monotonic(value_weeks: int = 0, sequence: str = DEFAULT_SEQUENCE) -> list[int]:
    """+1 on points, points per game and every weekly or quarter total; 0 elsewhere."""
    rising = {"points", "ppg", *WEEK_COLUMNS, *QUARTER_COLUMNS}
    return [1 if f in rising else 0 for f in feature_names(value_weeks, sequence)]


@dataclass
class WeeklyModel(WhatIfModelV2):
    """v2's learner on the weekly feature set. No points grid: monotone constraints carry the sweep."""

    value_weeks: int = 0
    sequence: str = DEFAULT_SEQUENCE
    feature_list: list[str] = field(default_factory=list)

    def fit(self, df: pd.DataFrame) -> "WeeklyModel":
        self.feature_list = feature_names(self.value_weeks, self.sequence)
        self.estimator = self._fit(df)
        self.train_seasons = sorted(int(s) for s in df["season"].unique())
        self._calibrate_band(df)
        return self

    def _fit(self, df: pd.DataFrame) -> HistGradientBoostingRegressor:
        estimator = HistGradientBoostingRegressor(monotonic_cst=monotonic(self.value_weeks, self.sequence), **HGB_PARAMS)
        return estimator.fit(
            make_features_weekly(df, self.value_weeks, self.sequence), np.log(df["end_value"] / df["start_value"])
        )

    def _calibrate_band(self, df: pd.DataFrame) -> None:
        last = max(self.train_seasons)
        earlier = df[df["season"] < last]
        if len(earlier):
            held_out = df[df["season"] == last]
            inner = WeeklyModel(
                label_source=self.label_source, scale_max=self.scale_max, value_weeks=self.value_weeks, sequence=self.sequence
            )
            inner.feature_list = self.feature_list
            inner.estimator = inner._fit(earlier)
            inner.train_seasons = sorted(int(s) for s in earlier["season"].unique())
            self.band_season = int(last)
        else:
            held_out, inner, self.band_season = df, self, None
        residuals = self._residuals(inner, held_out)
        self.band_offsets = {float(q): float(np.quantile(residuals, q)) for q in BAND_QUANTILES}
        self.band_n = len(residuals)

    @staticmethod
    def _residuals(model: "WeeklyModel", df: pd.DataFrame) -> np.ndarray:
        actual = np.log(df["end_value"].to_numpy(float) / df["start_value"].to_numpy(float))
        return actual - model.log_ratio(df)

    def log_ratio(self, df: pd.DataFrame) -> np.ndarray:
        if self.estimator is None:
            raise RuntimeError("Model is not fitted")
        return self.estimator.predict(make_features_weekly(df, self.value_weeks, self.sequence))

    def predict(self, df: pd.DataFrame) -> np.ndarray:
        pred = df["start_value"].to_numpy(float) * np.exp(self.log_ratio(df))
        return np.clip(pred, 0.0, self.scale_max)

    def predict_band(self, df: pd.DataFrame) -> dict[float, np.ndarray]:
        if not self.band_offsets:
            raise RuntimeError("Model has no calibrated band")
        log_ratio = self.log_ratio(df)
        start = df["start_value"].to_numpy(float)
        return {q: np.clip(start * np.exp(log_ratio + offset), 0.0, self.scale_max) for q, offset in self.band_offsets.items()}


def _skill(end, pred, start) -> float | None:
    base = _mae(end, start)
    return None if base == 0 else round(1 - _mae(end, pred) / base, 4)


def evaluate_weekly(
    df: pd.DataFrame,
    label_source: str,
    scale_max: float,
    min_train_seasons: int = 1,
    value_weeks: int = 0,
    sequence: str = DEFAULT_SEQUENCE,
) -> dict:
    """v2's folds for v2 and the weekly model; with ``value_weeks`` the weekly model also sees the value path."""
    rows, preds = [], []
    for test_season in season_folds(df, min_train_seasons):
        train = df[df["season"] < test_season]
        test = df[df["season"] == test_season].copy()
        v2 = WhatIfModelV2(label_source=label_source, scale_max=scale_max).fit(train)
        weekly = WeeklyModel(label_source=label_source, scale_max=scale_max, value_weeks=value_weeks, sequence=sequence).fit(
            train
        )
        test["pred_v2"] = v2.predict(test)
        test["pred_value"] = weekly.predict(test)
        band = weekly.predict_band(test)
        test["band_low"], test["band_high"] = band[min(band)], band[max(band)]
        end, start = test["end_value"].to_numpy(float), test["start_value"].to_numpy(float)
        rows.append(
            {
                "season": test_season,
                "n": len(test),
                "skill_v2": _skill(end, test["pred_v2"], start),
                "skill_weekly": _skill(end, test["pred_value"], start),
            }
        )
        preds.append(test)
    out = pd.concat(preds, ignore_index=True)
    end, start = out["end_value"].to_numpy(float), out["start_value"].to_numpy(float)
    return {
        "label_source": label_source,
        "model": f"weekly[{sequence}]" + (f"+value_path_{value_weeks}" if value_weeks else ""),
        "sequence": sequence,
        "features": feature_names(value_weeks, sequence),
        "folds": "train on seasons < Y, test on Y",
        "by_season": rows,
        "overall": {
            "skill_v2": _skill(end, out["pred_v2"], start),
            "skill_weekly": _skill(end, out["pred_value"], start),
            "weekly": summarize(out),
        },
        "by_position": {p: summarize(g) for p, g in out.groupby("position")},
        "band": {
            "quantiles": list(BAND_QUANTILES),
            "overall": band_summary(out),
            "by_season": {int(s): band_summary(g) for s, g in out.groupby("season")},
        },
        "predictions": out,
    }


def cmd_evaluate(args) -> None:
    source = get_label_source(args.labels, min_trades=args.min_trades, end_column=args.end_column)
    df = build_dataset_weekly(source, args.training_data) if args.training_data else build_dataset_weekly(source)
    if args.min_season:
        df = df[df["season"] >= args.min_season]
    if args.value_weeks:
        if args.value_path is None:
            raise SystemExit("--value-weeks needs --value-path, a CSV of player_id, season, week, value")
        df = attach_value_path(df, args.value_path, args.value_weeks)
    result = evaluate_weekly(df, source.name, source.scale_max, value_weeks=args.value_weeks, sequence=args.sequence)
    preds = result.pop("predictions")
    ARTIFACTS.mkdir(exist_ok=True)
    stem = f"{args.name or source.name}_weekly_{args.sequence.replace('+', '_')}" + (
        f"_value{args.value_weeks}" if args.value_weeks else ""
    )
    (ARTIFACTS / f"{stem}_eval.json").write_text(json.dumps(result, indent=2) + "\n")
    if args.save_predictions:
        preds.to_csv(ARTIFACTS / f"{stem}_predictions.csv", index=False)
    print(json.dumps({k: v for k, v in result.items() if k != "features"}, indent=2))


def attach_value_path(df: pd.DataFrame, path: Path | str, k: int) -> pd.DataFrame:
    """Join ``log(value in week w / start)`` for w <= k from a long CSV (player_id, season, week, value)."""
    values = pd.read_csv(path, dtype={"player_id": str})
    wide = values[values["week"].between(1, k)].pivot_table(index=["player_id", "season"], columns="week", values="value")
    wide.columns = [f"val_wk_{int(w)}" for w in wide.columns]
    out = df.merge(wide.reset_index(), on=["player_id", "season"], how="left")
    start = out["start_value"].to_numpy(float)
    for c in value_path_columns(k):
        if c not in out:
            out[c] = np.nan
        v = out[c].to_numpy(float)
        out[c] = np.where(np.isfinite(v) & (v > 0), np.log(np.where(v > 0, v, 1.0) / start), 0.0)
    return out


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(prog="predictor_v5.weekly_model")
    parser.add_argument("--labels", default="ktc", help="'ktc' or 'csv:<path>'")
    parser.add_argument("--min-trades", type=int, default=0, help="CSV labels: min trades at start and end")
    parser.add_argument("--end-column", default="end_value", help="CSV labels: column to use as end value")
    parser.add_argument("--training-data", help="stats zip (default data/training-data.zip)")
    parser.add_argument("--name", help="artifact name (default: the label source's)")
    sub = parser.add_subparsers(dest="cmd", required=True)
    ev = sub.add_parser("evaluate")
    ev.add_argument("--min-season", type=int, default=0, help="drop seasons before this one")
    ev.add_argument("--sequence", choices=sorted(SEQUENCES), default=DEFAULT_SEQUENCE)
    ev.add_argument("--value-weeks", type=int, default=0, help="also feed the value path through this week (mid-season task)")
    ev.add_argument("--value-path", help="CSV of player_id, season, week, value for --value-weeks")
    ev.add_argument("--save-predictions", action="store_true")
    ev.set_defaults(func=cmd_evaluate)
    args = parser.parse_args(argv)
    if args.labels == "ktc" and (args.min_trades or args.end_column != "end_value"):
        parser.error("--min-trades and --end-column only apply to CSV labels, not --labels ktc")
    args.func(args)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
