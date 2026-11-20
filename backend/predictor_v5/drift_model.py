"""Drift-split what-if model: a league-wide drift, then a relative move from points.

Every season the whole market moves. On the Backchannel/MFL labels the
median player loses about 10% of start value between the September start
and the January end whatever they do on the field; on KTC it is 1-4%. The
v1 and v2 models predict ``log(end / start)`` directly, so that slide is
folded into every prediction and the points signal has to carry it. Scored
against "no change" it also reads as skill, which flattered the source with
the bigger slide in the MFL-versus-KTC comparisons.

This model splits the two:

- **drift**: the season's median ``log(end / start)`` over the training
  seasons, one number per fitted model (``drift_how``: the mean of the
  training seasons' medians, or the last training season's).
- **relative**: the v2 gradient-boosted model (points, games, points per
  game, prior points, age, position, log start) fitted on
  ``log(end / start) - season median``, so it learns how a player moves
  against the league, not the league's own slide.

``end = start * exp(drift + relative)``. The band is calibrated on the same
held-out season as v2's, with residuals taken against drift plus relative.

``evaluate`` runs v2's chronological folds for both models and reports, per
test season, each one's skill against no change, the drift the split model
estimated, the median log change that season actually had, and an oracle
that replaces the estimated drift with the season's best single shift (the
median residual against the relative part): the ceiling a better drift
estimate could reach.

``model.py``, ``data.py``, ``evaluate.py`` and ``__main__.py`` are bound by
the sealed 2026 protocols; this module imports from them and changes nothing.

    cd backend
    python -m predictor_v5.drift_model evaluate                        # KTC folds
    python -m predictor_v5.drift_model --labels csv:predictor_v5/data/backchannel_labels_2019_2025_mfl_full.csv \\
        --min-trades 5 evaluate --drift mean
"""

import argparse
import json
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingRegressor

from predictor_v5.evaluate import _mae, season_folds, summarize
from predictor_v5.labels import get_label_source
from predictor_v5.model import HGB_PARAMS
from predictor_v5.whatif_v2 import (
    BAND_QUANTILES,
    MONOTONIC_V2,
    WhatIfModelV2,
    band_summary,
    build_dataset_v2,
    make_features_v2,
)

ARTIFACTS = Path(__file__).resolve().parent / "artifacts"
DRIFT_HOW = ("mean", "last")


def log_change(df: pd.DataFrame) -> np.ndarray:
    return np.log(df["end_value"].to_numpy(float) / df["start_value"].to_numpy(float))


def season_drifts(df: pd.DataFrame) -> dict[int, float]:
    """Each season's median ``log(end / start)``: the league-wide move a typical player made."""
    change = pd.Series(log_change(df), index=df.index)
    return {int(s): float(change[idx].median()) for s, idx in df.groupby("season").groups.items()}


def drift_estimate(drifts: dict[int, float], how: str = "mean") -> float:
    """One drift for the next season from the training seasons' medians."""
    if not drifts:
        raise ValueError("no training seasons")
    if how == "mean":
        return float(np.mean(list(drifts.values())))
    if how == "last":
        return drifts[max(drifts)]
    raise ValueError(f"drift_how must be one of {DRIFT_HOW}, not {how!r}")


@dataclass
class DriftSplitModel(WhatIfModelV2):
    """v2's model on the de-drifted target, plus one drift per fitted model."""

    drift_how: str = "mean"
    season_drift: dict[int, float] = field(default_factory=dict)
    drift: float = 0.0

    def fit(self, df: pd.DataFrame) -> "DriftSplitModel":
        self.season_drift = season_drifts(df)
        self.drift = drift_estimate(self.season_drift, self.drift_how)
        self.estimator = self._fit_estimator(df)
        self.train_seasons = sorted(self.season_drift)
        self._calibrate_band(df)
        return self

    @staticmethod
    def _fit_estimator(df: pd.DataFrame) -> HistGradientBoostingRegressor:
        drifts = season_drifts(df)
        relative = log_change(df) - df["season"].astype(int).map(drifts).to_numpy(float)
        estimator = HistGradientBoostingRegressor(monotonic_cst=MONOTONIC_V2, **HGB_PARAMS)
        return estimator.fit(make_features_v2(df), relative)

    def _calibrate_band(self, df: pd.DataFrame) -> None:
        """As v2: residuals on the last training season from a model fitted on the earlier ones."""
        last = max(self.train_seasons)
        earlier = df[df["season"] < last]
        if len(earlier):
            held_out = df[df["season"] == last]
            inner = DriftSplitModel(label_source=self.label_source, scale_max=self.scale_max, drift_how=self.drift_how)
            inner.season_drift = season_drifts(earlier)
            inner.drift = drift_estimate(inner.season_drift, self.drift_how)
            inner.estimator = self._fit_estimator(earlier)
            inner.train_seasons = sorted(inner.season_drift)
            self.band_season = int(last)
        else:
            held_out, inner, self.band_season = df, self, None
        residuals = self._log_residuals(inner, held_out)
        self.band_offsets = {float(q): float(np.quantile(residuals, q)) for q in BAND_QUANTILES}
        self.band_n = len(residuals)

    @staticmethod
    def _log_residuals(model: "DriftSplitModel", df: pd.DataFrame) -> np.ndarray:
        return log_change(df) - (model.drift + model._monotone_log_ratio(make_features_v2(df)))

    def predict_parts(self, df: pd.DataFrame) -> dict[str, np.ndarray]:
        """The two pieces of every prediction, in log ratio: the league's drift and the player's relative move."""
        if self.estimator is None:
            raise RuntimeError("Model is not fitted")
        relative = self._monotone_log_ratio(make_features_v2(df))
        return {"drift": np.full(len(df), self.drift), "relative": relative}

    def predict(self, df: pd.DataFrame) -> np.ndarray:
        parts = self.predict_parts(df)
        pred = df["start_value"].to_numpy(float) * np.exp(parts["drift"] + parts["relative"])
        return np.clip(pred, 0.0, self.scale_max)

    def predict_band(self, df: pd.DataFrame) -> dict[float, np.ndarray]:
        if not self.band_offsets:
            raise RuntimeError("Model has no calibrated band")
        parts = self.predict_parts(df)
        start = df["start_value"].to_numpy(float)
        log_ratio = parts["drift"] + parts["relative"]
        return {q: np.clip(start * np.exp(log_ratio + offset), 0.0, self.scale_max) for q, offset in self.band_offsets.items()}


def _skill(end, pred, start) -> float | None:
    base = _mae(end, start)
    return None if base == 0 else 1 - _mae(end, pred) / base


def evaluate_drift(
    df: pd.DataFrame, label_source: str, scale_max: float, min_train_seasons: int = 1, drift_how: str = "mean"
) -> dict:
    """v2's folds for v2 and the drift-split model, with the drift each season had and the oracle ceiling."""
    rows, preds = [], []
    for test_season in season_folds(df, min_train_seasons):
        train = df[df["season"] < test_season]
        test = df[df["season"] == test_season].copy()
        v2 = WhatIfModelV2(label_source=label_source, scale_max=scale_max).fit(train)
        split = DriftSplitModel(label_source=label_source, scale_max=scale_max, drift_how=drift_how).fit(train)
        test["pred_v2"] = v2.predict(test)
        test["pred_value"] = split.predict(test)
        parts = split.predict_parts(test)
        actual_drift = float(np.median(log_change(test)))
        # The best single shift for this season given the relative part: the median residual
        # against relative alone. The season's median log change is not it, because the
        # relative part already carries the median player's move from points.
        oracle_shift = float(np.median(log_change(test) - parts["relative"]))
        start = test["start_value"].to_numpy(float)
        test["pred_oracle"] = np.clip(start * np.exp(oracle_shift + parts["relative"]), 0.0, scale_max)
        band = split.predict_band(test)
        test["band_low"], test["band_high"] = band[min(band)], band[max(band)]
        end = test["end_value"].to_numpy(float)
        rows.append(
            {
                "season": test_season,
                "n": len(test),
                "drift_estimated": round(split.drift, 4),
                "drift_actual": round(actual_drift, 4),
                "oracle_shift": round(oracle_shift, 4),
                "skill_v2": _round(_skill(end, test["pred_v2"], start)),
                "skill_drift_split": _round(_skill(end, test["pred_value"], start)),
                "skill_oracle_drift": _round(_skill(end, test["pred_oracle"], start)),
            }
        )
        preds.append(test)
    out = pd.concat(preds, ignore_index=True)
    end, start = out["end_value"].to_numpy(float), out["start_value"].to_numpy(float)
    return {
        "label_source": label_source,
        "model": "drift_split",
        "drift_how": drift_how,
        "folds": "train on seasons < Y, test on Y",
        "by_season": rows,
        "overall": {
            "skill_v2": _round(_skill(end, out["pred_v2"], start)),
            "skill_drift_split": _round(_skill(end, out["pred_value"], start)),
            "skill_oracle_drift": _round(_skill(end, out["pred_oracle"], start)),
            "drift_split": summarize(out),
        },
        "by_position": {p: summarize(g) for p, g in out.groupby("position")},
        "band": {
            "quantiles": list(BAND_QUANTILES),
            "overall": band_summary(out),
            "by_season": {int(s): band_summary(g) for s, g in out.groupby("season")},
        },
        "predictions": out,
    }


def _round(x: float | None) -> float | None:
    return None if x is None else round(float(x), 4)


def cmd_evaluate(args) -> None:
    source = get_label_source(args.labels, min_trades=args.min_trades, end_column=args.end_column)
    df = build_dataset_v2(source, args.training_data) if args.training_data else build_dataset_v2(source)
    if args.min_season:
        df = df[df["season"] >= args.min_season]
    result = evaluate_drift(df, source.name, source.scale_max, drift_how=args.drift)
    preds = result.pop("predictions")
    ARTIFACTS.mkdir(exist_ok=True)
    stem = f"{args.name or source.name}_drift_split_{args.drift}"
    (ARTIFACTS / f"{stem}_eval.json").write_text(json.dumps(result, indent=2) + "\n")
    if args.save_predictions:
        preds.to_csv(ARTIFACTS / f"{stem}_predictions.csv", index=False)
    print(json.dumps(result, indent=2))


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(prog="predictor_v5.drift_model")
    parser.add_argument("--labels", default="ktc", help="'ktc' or 'csv:<path>'")
    parser.add_argument("--min-trades", type=int, default=0, help="CSV labels: min trades at start and end")
    parser.add_argument("--end-column", default="end_value", help="CSV labels: column to use as end value")
    parser.add_argument("--training-data", help="stats zip (default data/training-data.zip)")
    parser.add_argument("--name", help="artifact name (default: the label source's)")
    sub = parser.add_subparsers(dest="cmd", required=True)
    ev = sub.add_parser("evaluate")
    ev.add_argument("--drift", choices=DRIFT_HOW, default="mean")
    ev.add_argument("--min-season", type=int, default=0, help="drop seasons before this one")
    ev.add_argument("--save-predictions", action="store_true")
    ev.set_defaults(func=cmd_evaluate)
    args = parser.parse_args(argv)
    if args.labels == "ktc" and (args.min_trades or args.end_column != "end_value"):
        parser.error("--min-trades and --end-column only apply to CSV labels, not --labels ktc")
    args.func(args)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
