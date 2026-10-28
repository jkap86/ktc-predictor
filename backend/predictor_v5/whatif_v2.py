"""What-if model v2: points, games played and prior-year points.

The v1 model (``model.py``) sees a season as one points total. The market
prices per-game production, so a 200-point season in 12 games moves value
more than 200 in 17. v2 adds games played, points per game and the prior
season's points to the same gradient-boosted log-ratio model. On the KTC
season folds (train < Y, test Y, 2021-25) skill versus no-change rises from
0.275 to about 0.34; on Backchannel labels from 0.44 to 0.47.

``model.py``, ``data.py``, ``evaluate.py`` and ``__main__.py`` are bound by
the sealed 2026 confirmation and replacement protocols (their SHA-256s sit
in the manifests), so this module imports from them and changes nothing in
them. Those protocols keep scoring the v1 model; v2 is for the season tool
and new evaluations.

    cd backend
    python -m predictor_v5.whatif_v2 evaluate                       # KTC folds
    python -m predictor_v5.whatif_v2 --labels backchannel:predictor_v5/data/backchannel_labels_2019_2025.csv \\
        --min-trades 5 evaluate
    python -m predictor_v5.whatif_v2 season --player "Bijan Robinson" --points 300 --games 16
"""

import argparse
import json
import sys
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingRegressor

from predictor_v5.data import DEFAULT_ZIP, POSITIONS, build_dataset, drop_pre_nfl, read_training_players
from predictor_v5.evaluate import _mae, season_folds, summarize
from predictor_v5.labels import LabelSource, get_label_source
from predictor_v5.model import HGB_PARAMS, POINTS_GRID_STEP, WhatIfModel

ARTIFACTS = Path(__file__).resolve().parent / "artifacts"

FEATURES_V2 = ["log_start", "age", "points", "games", "ppg", "prior_points", "position_code"]
# +1: monotonic increasing; 0: unconstrained. Points and points per game
# both rise together when games is held fixed, so the what-if curve stays
# monotone in points.
MONOTONIC_V2 = [0, 0, 1, 0, 1, 0, 0]
# A full regular season, and the default when a caller gives only points.
FULL_SEASON_GAMES = 17


def points_per_game(points, games) -> np.ndarray:
    """Points per game; 0 when no games were played (inactive seasons are 0 points anyway)."""
    points = np.asarray(points, float)
    games = np.asarray(games, float)
    return np.where(games > 0, points / np.maximum(games, 1.0), 0.0)


def read_season_stats_v2(zip_path: Path | str = DEFAULT_ZIP) -> pd.DataFrame:
    """One row per (player_id, season) with v1's columns plus ``prior_points``.

    ``prior_points`` is the player's points in the previous season. It comes
    from that season's own row in the archive when there is one (the same
    points definition as ``points``, and the same source the season tool
    uses for a new season), otherwise from the archive's ``prior_year_fp``
    field, which covers the season before the archive starts (2020 rows). A
    player with neither, such as a rookie, gets 0.
    """
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
                    "prior_year_fp": float(s.get("prior_year_fp") or 0.0),
                }
            )
    stats = pd.DataFrame(rows)
    previous = stats[["player_id", "season", "points"]].assign(season=stats["season"] + 1)
    previous = previous.rename(columns={"points": "prior_points"})
    stats = stats.merge(previous, on=["player_id", "season"], how="left")
    stats["prior_points"] = stats["prior_points"].fillna(stats["prior_year_fp"]).astype(float)
    return stats.drop(columns=["prior_year_fp"])


def load_season_stats_v2(zip_path: Path | str = DEFAULT_ZIP) -> pd.DataFrame:
    """``read_season_stats_v2`` without college (devy) seasons."""
    return drop_pre_nfl(read_season_stats_v2(zip_path))


def prior_points_for(season: int, stats: pd.DataFrame | None = None) -> dict[str, float]:
    """Each player's points in ``season - 1``, from the stats archive (0 when absent)."""
    stats = read_season_stats_v2() if stats is None else stats
    last = stats[stats["season"] == season - 1]
    return {str(pid): float(pts) for pid, pts in zip(last["player_id"], last["points"])}


def make_features_v2(df: pd.DataFrame) -> np.ndarray:
    unknown = set(df["position"]) - set(POSITIONS)
    if unknown:
        raise ValueError(f"Unknown positions: {sorted(unknown)}")
    points = df["points"].astype(float).to_numpy()
    games = df["games"].astype(float).to_numpy()
    return np.column_stack(
        [
            np.log(df["start_value"].astype(float)),
            df["age"].astype(float),
            points,
            games,
            points_per_game(points, games),
            df["prior_points"].astype(float),
            df["position"].map(POSITIONS.index).astype(float),
        ]
    )


@dataclass
class WhatIfModelV2(WhatIfModel):
    """v1's model with the v2 feature set. Same target, loss and parameters."""

    def fit(self, df: pd.DataFrame) -> "WhatIfModelV2":
        self.estimator = HistGradientBoostingRegressor(monotonic_cst=MONOTONIC_V2, **HGB_PARAMS)
        y = np.log(df["end_value"] / df["start_value"])
        self.estimator.fit(make_features_v2(df), y)
        self.train_seasons = sorted(int(s) for s in df["season"].unique())
        return self

    def predict(self, df: pd.DataFrame) -> np.ndarray:
        if self.estimator is None:
            raise RuntimeError("Model is not fitted")
        log_ratio = self._monotone_log_ratio(make_features_v2(df))
        pred = df["start_value"].to_numpy(float) * np.exp(log_ratio)
        return np.clip(pred, 0.0, self.scale_max)

    def _monotone_log_ratio(self, X: np.ndarray) -> np.ndarray:
        """Running max over a points grid, with points per game moving along it.

        Like v1, but each grid point sets both ``points`` and ``ppg`` (at the
        row's games), so the curve a caller sweeps is the one made monotone.
        """
        pts_col, ppg_col, games_col = (FEATURES_V2.index(c) for c in ("points", "ppg", "games"))
        idx = np.floor(np.maximum(X[:, pts_col], 0.0) / POINTS_GRID_STEP).astype(int)
        grid = np.arange(idx.max() + 1) * POINTS_GRID_STEP
        others = np.delete(X, [pts_col, ppg_col], axis=1)
        base, inverse = np.unique(others, axis=0, return_inverse=True)
        n_base, n_grid = len(base), len(grid)
        games = np.delete(np.arange(X.shape[1]), [pts_col, ppg_col]).tolist().index(games_col)
        grid_points = np.tile(grid, n_base)
        grid_games = np.repeat(base[:, games], n_grid)
        Xg = np.repeat(base, n_grid, axis=0)
        Xg = np.insert(Xg, pts_col, grid_points, axis=1)
        Xg = np.insert(Xg, ppg_col, points_per_game(grid_points, grid_games), axis=1)
        on_grid = self.estimator.predict(Xg).reshape(n_base, n_grid)
        running_max = np.maximum.accumulate(on_grid, axis=1)
        return running_max[inverse.ravel(), idx]

    def predict_one(
        self,
        *,
        start_value: float,
        age: float,
        position: str,
        points: float,
        games: float = FULL_SEASON_GAMES,
        prior_points: float = 0.0,
    ) -> float:
        if start_value <= 0:
            raise ValueError("start_value must be > 0")
        if games < 0:
            raise ValueError("games must be >= 0")
        row = pd.DataFrame(
            [
                {
                    "start_value": start_value,
                    "age": age,
                    "position": position.upper(),
                    "points": points,
                    "games": games,
                    "prior_points": prior_points,
                }
            ]
        )
        return float(self.predict(row)[0])


def build_dataset_v2(labels: LabelSource, zip_path: Path | str = DEFAULT_ZIP) -> pd.DataFrame:
    return build_dataset(load_season_stats_v2(zip_path), labels)


def evaluate_v2(df: pd.DataFrame, label_source: str, scale_max: float, min_train_seasons: int = 1) -> dict:
    """The same chronological folds as ``evaluate.evaluate`` with the v2 model."""
    preds = []
    for test_season in season_folds(df, min_train_seasons):
        train = df[df["season"] < test_season]
        test = df[df["season"] == test_season].copy()
        model = WhatIfModelV2(label_source=label_source, scale_max=scale_max).fit(train)
        test["pred_value"] = model.predict(test)
        preds.append(test)
    out = pd.concat(preds, ignore_index=True)
    return {
        "label_source": label_source,
        "model": "whatif_v2",
        "features": list(FEATURES_V2),
        "folds": "train on seasons < Y, test on Y",
        "overall": summarize(out),
        "by_season": {int(s): summarize(g) for s, g in out.groupby("season")},
        "by_position": {p: summarize(g) for p, g in out.groupby("position")},
        "beats_no_change": bool(_mae(out["end_value"], out["pred_value"]) < _mae(out["end_value"], out["start_value"])),
        "predictions": out,
    }


def cmd_evaluate(args) -> None:
    source = get_label_source(args.labels, min_trades=args.min_trades, end_column=args.end_column)
    df = build_dataset_v2(source)
    if args.min_season:
        df = df[df["season"] >= args.min_season]
    result = evaluate_v2(df, source.name, source.scale_max)
    preds = result.pop("predictions")
    ARTIFACTS.mkdir(exist_ok=True)
    (ARTIFACTS / f"{source.name}_v2_eval.json").write_text(json.dumps(result, indent=2) + "\n")
    if args.save_predictions:
        preds.to_csv(ARTIFACTS / f"{source.name}_v2_eval_predictions.csv", index=False)
    print(json.dumps(result, indent=2))


def cmd_season(args) -> None:
    from predictor_v5.season import SOURCES, SeasonTool

    tool = SeasonTool(season=args.season)
    matches = tool.find(args.player)
    if len(matches) != 1:
        print(f"{len(matches)} players match {args.player!r}" + (":" if matches else "."))
        for m in matches:
            print(f"  {m['player_id']:>6}  {m['name']} ({m['position']}, {m['team'] or 'FA'})")
        return
    points = [float(p) for p in args.sweep.split(",")] if args.sweep else [args.points]
    out = tool.predict(matches[0]["player_id"], points, age=args.age, games=args.games)
    age = f"age {out['age']:g}" if out["age"] is not None else "age unknown"
    print(
        f"{out['name']} ({out['position']}, {out['team'] or 'FA'}, {age}), {out['season']} season, "
        f"{out['games']:g} games, {out['prior_points']:g} pts in {out['season'] - 1}"
    )
    for name, cfg in SOURCES.items():
        src = out[name]
        if not src["predictions"]:
            print(f"  {cfg['label']}: {src['note']}")
            continue
        print(f"  {cfg['label']}: start {src['start_value']:.0f} on {src['start_date']}")
        for p in src["predictions"]:
            print(f"    {p['points']:g} pts -> {p['value']} ({p['change']:+d})")


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(prog="predictor_v5.whatif_v2")
    parser.add_argument("--labels", default="ktc", help="'ktc' or 'csv:<path>'")
    parser.add_argument("--min-trades", type=int, default=0, help="CSV labels: min trades at start and end")
    parser.add_argument("--end-column", default="end_value", help="CSV labels: column to use as end value")
    sub = parser.add_subparsers(dest="cmd", required=True)

    ev = sub.add_parser("evaluate")
    ev.add_argument("--min-season", type=int, default=0, help="drop seasons before this one")
    ev.add_argument("--save-predictions", action="store_true")
    ev.set_defaults(func=cmd_evaluate)

    se = sub.add_parser("season")
    se.add_argument("--player", required=True, help="player id, exact name, or name substring")
    se.add_argument("--points", type=float, default=250.0)
    se.add_argument("--games", type=float, default=FULL_SEASON_GAMES)
    se.add_argument("--sweep", help="comma-separated points totals")
    se.add_argument("--age", type=float)
    se.add_argument("--season", type=int, default=2026)
    se.set_defaults(func=cmd_season)

    args = parser.parse_args(argv)
    if args.labels == "ktc" and (args.min_trades or args.end_column != "end_value"):
        parser.error("--min-trades and --end-column only apply to CSV labels, not --labels ktc")
    args.func(args)
    return 0


if __name__ == "__main__":
    sys.exit(main())
