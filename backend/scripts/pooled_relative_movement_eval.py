"""Option B check (2026-10-28): pooled MFL/Backchannel + KTC relative-movement model vs v2 per source.

Run: cd backend && python scripts/pooled_relative_movement_eval.py
Writes nothing into the package; results are in the predictor_v5 README section
"Pooled MFL + KTC relative-movement model does not beat v2".
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingRegressor
from predictor_v5.data import POSITIONS
from predictor_v5.labels import KtcLabels, CsvLabels
from predictor_v5.whatif_v2 import build_dataset_v2, WhatIfModelV2, points_per_game
from predictor_v5.model import HGB_PARAMS

MFL = str(Path(__file__).resolve().parents[1] / "predictor_v5/data/backchannel_labels_2019_2025_mfl.csv")

def load():
    k = build_dataset_v2(KtcLabels()); k["source"] = "ktc"; k["source_code"] = 0.0
    b = build_dataset_v2(CsvLabels(MFL, name="backchannel", min_trades=5)); b["source"] = "backchannel"; b["source_code"] = 1.0
    df = pd.concat([k, b], ignore_index=True)
    df["log_ratio"] = np.log(df.end_value / df.start_value)
    g = df.groupby(["source", "season"])
    df["season_median"] = g["log_ratio"].transform("median")
    df["rel"] = df.log_ratio - df.season_median
    df["start_pct"] = g["start_value"].rank(pct=True)
    # star-relative: log(start / season-source 90th percentile)
    p90 = g["start_value"].transform(lambda s: s.quantile(0.9))
    df["start_rel90"] = np.log(df.start_value / p90)
    df["log_start"] = np.log(df.start_value)
    df["ppg"] = points_per_game(df.points, df.games)
    df["position_code"] = df.position.map(POSITIONS.index).astype(float)
    return df

def feats(df, scale, flag):
    cols = [scale, "age", "points", "games", "ppg", "prior_points", "position_code"] + (["source_code"] if flag else [])
    mono = [0, 0, 1, 0, 1, 0, 0] + ([0] if flag else [])
    return df[cols].to_numpy(float), mono

def fit_rel(train, scale, flag):
    X, mono = feats(train, scale, flag)
    m = HistGradientBoostingRegressor(monotonic_cst=mono, **HGB_PARAMS).fit(X, train.rel)
    return m

def drift_estimates(train, test_season):
    """Per-source drift for the test season: mean of training-season medians, and last season's median."""
    med = train.groupby(["source", "season"]).log_ratio.median().reset_index()
    out = {}
    for src, g in med.groupby("source"):
        out[src] = {"mean": float(g.log_ratio.mean()), "last": float(g[g.season == g.season.max()].log_ratio.iloc[0]),
                    "mean3": float(g.sort_values("season").log_ratio.tail(3).mean())}
    return out

def mae(a, b): return float(np.mean(np.abs(np.asarray(a) - np.asarray(b))))

df = load()
print(df.groupby(["source", "season"]).size().unstack(0))
print("season medians:\n", df.groupby(["source", "season"]).log_ratio.median().unstack(0).round(3))

rows = []
for test_season in (2022, 2023, 2024, 2025):
    train = df[df.season < test_season]; test = df[df.season == test_season].copy()
    drifts = drift_estimates(train, test_season)
    # v2 single-source baselines
    for src in ("ktc", "backchannel"):
        tr, te = train[train.source == src], test[test.source == src].copy()
        v2 = WhatIfModelV2(label_source=src, scale_max=10000.0).fit(tr)
        te["v2"] = v2.predict(te)
        test.loc[te.index, "v2"] = te["v2"]
    for scale in ("start_pct", "start_rel90", "log_start"):
        for flag in (True, False):
            for pool in ("pooled", "single"):
                for drift_kind in ("mean", "mean3", "last", "oracle"):
                    for src in ("ktc", "backchannel"):
                        te = test[test.source == src].copy()
                        tr = train if pool == "pooled" else train[train.source == src]
                        if pool == "single" and flag: continue
                        m = fit_rel(tr, scale, flag)
                        X, _ = feats(te, scale, flag)
                        rel_hat = m.predict(X)
                        drift = te.season_median.iloc[0] if drift_kind == "oracle" else drifts[src][drift_kind]
                        pred = np.clip(te.start_value * np.exp(rel_hat + drift), 0, 10000)
                        rows.append(dict(test=test_season, scale=scale, flag=flag, pool=pool, drift=drift_kind, source=src, n=len(te),
                                         mae=mae(te.end_value, pred), v2_mae=mae(te.end_value, te.v2), nochange=mae(te.end_value, te.start_value),
                                         rel_logmae=mae(te.rel, rel_hat), rel_nochange=mae(te.rel, 0)))
res = pd.DataFrame(rows)
res.to_csv(Path(__file__).resolve().parent / "pooled_relative_movement_results.csv", index=False)
# aggregate: pooled MAE across folds (sum of abs errors / n) vs v2
agg = res.assign(ae=res.mae * res.n, v2ae=res.v2_mae * res.n, ncae=res.nochange * res.n, rae=res.rel_logmae * res.n).groupby(
    ["scale", "flag", "pool", "drift", "source"]).agg(n=("n", "sum"), ae=("ae", "sum"), v2ae=("v2ae", "sum"), ncae=("ncae", "sum"), rae=("rae", "sum"))
agg["mae"] = agg.ae / agg.n; agg["v2"] = agg.v2ae / agg.n; agg["nc"] = agg.ncae / agg.n; agg["skill"] = 1 - agg.mae / agg.nc; agg["v2skill"] = 1 - agg.v2 / agg.nc
agg["rel_logmae"] = agg.rae / agg.n
pd.set_option("display.width", 200); pd.set_option("display.max_rows", 500)
print(agg[["n", "mae", "v2", "nc", "skill", "v2skill", "rel_logmae"]].round(3).to_string())
