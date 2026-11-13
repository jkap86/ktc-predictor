"""Tuning sweep for WhatIfModelV2 (2026-10-29): select on the 2022-24 folds of both
sources, report 2025 separately. Result: nothing beats the current parameters by
more than 0.3 skill points; see the predictor_v5 README section "Tuning v2".

    cd backend && python scripts/tune_whatif_v2.py 1        # 81 tree-parameter combinations
    cd backend && python scripts/tune_whatif_v2.py 2 '{}'   # l2, loss, window, per-position, depth
"""
import sys, itertools, json, time
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import numpy as np, pandas as pd
from sklearn.ensemble import HistGradientBoostingRegressor
from predictor_v5.labels import KtcLabels, CsvLabels
from predictor_v5 import whatif_v2 as v2
from predictor_v5 import model as M

MFL = str(Path(__file__).resolve().parents[1] / "predictor_v5/data/backchannel_labels_2019_2025_mfl.csv")
OUT = str(Path(__file__).resolve().parent / "tune_whatif_v2_results.csv")
stage = sys.argv[1] if len(sys.argv) > 1 else "1"

data = {
    "ktc": v2.build_dataset_v2(KtcLabels()),
    "backchannel": v2.build_dataset_v2(CsvLabels(MFL, name="backchannel", min_trades=5)),
}
SCALE = {"ktc": 9999.0, "backchannel": 10000.0}
SELECT = (2022, 2023, 2024); HOLD = 2025


def fit_predict(train, test, params, per_position, scale_max):
    def one(tr, te):
        m = v2.WhatIfModelV2(label_source="x", scale_max=scale_max)
        m.estimator = HistGradientBoostingRegressor(monotonic_cst=v2.MONOTONIC_V2, **params)
        y = np.log(tr.end_value / tr.start_value)
        m.estimator.fit(v2.make_features_v2(tr), y)
        return m.predict(te)
    if not per_position:
        return one(train, test)
    out = np.empty(len(test))
    for pos in test.position.unique():
        mask = (test.position == pos).to_numpy()
        out[mask] = one(train[train.position == pos], test[mask])
    return out


def run(cfg):
    params = dict(M.HGB_PARAMS); params.update(cfg["params"])
    rec = {"name": cfg["name"]}
    rec.update({f"p_{k}": v for k, v in params.items()})
    rec["window"] = cfg.get("window", 0); rec["per_position"] = cfg.get("per_position", False)
    for src, df in data.items():
        for season in SELECT + (HOLD,):
            train = df[df.season < season]
            if cfg.get("window"):
                train = train[train.season >= season - cfg["window"]]
            test = df[df.season == season]
            pred = fit_predict(train, test, params, cfg.get("per_position", False), SCALE[src])
            rec[f"ae_{src}_{season}"] = float(np.abs(test.end_value - pred).sum())
            rec[f"nc_{src}_{season}"] = float(np.abs(test.end_value - test.start_value).sum())
            rec[f"n_{src}_{season}"] = len(test)
    for src in data:
        ae = sum(rec[f"ae_{src}_{s}"] for s in SELECT); nc = sum(rec[f"nc_{src}_{s}"] for s in SELECT)
        rec[f"skill_sel_{src}"] = 1 - ae / nc
        rec[f"skill_2025_{src}"] = 1 - rec[f"ae_{src}_2025"] / rec[f"nc_{src}_2025"]
    rec["skill_sel"] = (rec["skill_sel_ktc"] + rec["skill_sel_backchannel"]) / 2
    rec["skill_2025"] = (rec["skill_2025_ktc"] + rec["skill_2025_backchannel"]) / 2
    return rec


cfgs = [{"name": "base", "params": {}}]
if stage == "1":
    for lr, it, leaves, msl in itertools.product((0.03, 0.05, 0.1), (150, 300, 600), (7, 15, 31), (10, 20, 40)):
        cfgs.append({"name": f"lr{lr}_it{it}_lf{leaves}_msl{msl}",
                     "params": {"learning_rate": lr, "max_iter": it, "max_leaf_nodes": leaves, "min_samples_leaf": msl}})
else:
    best = json.loads(sys.argv[2]) if len(sys.argv) > 2 else {}
    for l2 in (0.0, 1.0, 5.0, 20.0):
        for loss in ("absolute_error", "squared_error"):
            for window in (0, 3, 4):
                for pp in (False, True):
                    p = dict(best); p.update({"l2_regularization": l2, "loss": loss})
                    cfgs.append({"name": f"l2{l2}_{loss[:3]}_w{window}_pp{int(pp)}", "params": p, "window": window, "per_position": pp})
    for depth in (3, 4, 6):
        p = dict(best); p.update({"max_depth": depth})
        cfgs.append({"name": f"best_depth{depth}", "params": p})

rows = []; t0 = time.time()
for i, cfg in enumerate(cfgs):
    rows.append(run(cfg))
    if i % 20 == 0:
        print(f"{i}/{len(cfgs)} {time.time()-t0:.0f}s", flush=True)
res = pd.DataFrame(rows); res.to_csv(OUT if stage == "1" else OUT.replace(".csv", "_2.csv"), index=False)
pd.set_option("display.width", 250); pd.set_option("display.max_rows", 60)
cols = ["name", "skill_sel_ktc", "skill_sel_backchannel", "skill_sel", "skill_2025_ktc", "skill_2025_backchannel", "skill_2025"]
print(res.sort_values("skill_sel", ascending=False)[cols].head(25).round(4).to_string())
print("base:\n", res[res.name == "base"][cols].round(4).to_string())
