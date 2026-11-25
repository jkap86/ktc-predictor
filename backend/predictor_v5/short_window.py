"""Short-window what-if study: value change over the next 3 or 6 games.

The season tool (``whatif_v2``) predicts week 1 to the end of the regular
season. This module asks whether the same idea works over a short window:
given a player's value on the Tuesday before week ``w``, their age,
position, recent form and the points they score in weeks ``w .. w+k-1`` (the
user's what-if), predict their value on the Tuesday after week ``w+k-1``.

Labels, both on Tuesdays (after Monday night, before the next week):

- **Backchannel** (``--source bc``): ``sql/weekly_values.sql``, the
  ``baseline-hl28-v1`` dynasty superflex board's stored reference-scale
  value at each week, newest revision. ``--min-trades 5`` keeps windows
  whose start and end values each rest on 5+ trades in the 150-day window.
- **KTC** (``--source ktc``): the training zip's ``weekly_ktc`` (week ``w`` is
  the Tuesday before week ``w``). Weeks 1-18 only, so KTC windows end by the
  Tuesday before week 18.

Every start week is used, so windows overlap; each fold trains on seasons
before the test season and is scored on that season's windows. Folds
2022-24 are selection, 2025 is the check. Nothing here touches the sealed
modules or the season tool.

    cd backend
    python -m predictor_v5.short_window evaluate --source bc --values <weekly_values.csv[.gz]>
    python -m predictor_v5.short_window evaluate --source ktc
"""

import argparse
import json
import zipfile
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingRegressor

from predictor_v5.data import POSITIONS
from predictor_v5.model import HGB_PARAMS

HERE = Path(__file__).resolve().parent
ZIP = HERE.parent / "data" / "training-data-2019.zip"
DEFAULT_VALUES = HERE / "data" / "weekly_values_v1_sf.csv.gz"
ARTIFACTS = HERE / "artifacts"
SELECTION = (2022, 2023, 2024)
CHECK = 2025
RECENT = 3

FEATURES = {
    "full": [
        "log_start", "age", "position_code", "week",
        "win_points", "win_games", "win_ppg",
        "std_ppg", "std_games", "recent_ppg", "prior_points", "momentum",
    ],
}
FEATURES["no_window_points"] = [f for f in FEATURES["full"] if not f.startswith("win_")]
FEATURES["no_momentum"] = [f for f in FEATURES["full"] if f != "momentum"]
FEATURES["points_only"] = ["log_start", "age", "position_code", "win_points", "win_games", "win_ppg"]
RISING = {"win_points", "win_ppg"}
SEASON_FEATURES = ["log_start", "age", "points", "games", "ppg", "prior_points", "position_code"]


def regular_weeks(season: int) -> int:
    return 17 if season <= 2020 else 18


# ---------------------------------------------------------------- data


def read_zip(path: Path = ZIP) -> tuple[pd.DataFrame, pd.DataFrame]:
    """(weekly points, weekly KTC): one row per player-season-week."""
    with zipfile.ZipFile(path) as z:
        players = json.load(z.open("training-data.json"))["players"]
    pts, ktc = [], []
    for p in players:
        if p["position"] not in POSITIONS:
            continue
        pid = str(p["player_id"])
        for s in p.get("seasons", []):
            year = int(s["year"])
            if int(s.get("years_exp") or 0) == 0 and not int(s.get("games_played") or 0):
                continue  # college (devy) season, as drop_pre_nfl
            base = {"player_id": pid, "name": p.get("name"), "position": p["position"], "season": year,
                    "age": float(s["age"]), "season_points": float(s.get("fantasy_points") or 0.0),
                    "season_games": int(s.get("games_played") or 0)}
            for w in s.get("weekly_stats") or []:
                pts.append({**base, "week": int(w["week"]), "points": float(w.get("fantasy_points") or 0.0),
                            "played": int(bool(w.get("games_played")))})
            for w in s.get("weekly_ktc") or []:
                if w.get("ktc"):
                    ktc.append({"player_id": pid, "season": year, "week": int(w["week"]), "value": float(w["ktc"]),
                                "trades_window": np.nan})
    pts = pd.DataFrame(pts)
    season = pts.drop_duplicates(["player_id", "season"])[["player_id", "season", "season_points"]]
    prior = season.assign(season=season["season"] + 1).rename(columns={"season_points": "prior_points"})
    pts = pts.merge(prior, on=["player_id", "season"], how="left")
    pts["prior_points"] = pts["prior_points"].fillna(0.0)
    return pts, pd.DataFrame(ktc)


def read_bc_values(path: Path) -> pd.DataFrame:
    v = pd.read_csv(path, dtype={"player_id": str})
    v = v[~v["player_id"].str.contains(r"\|")]  # draft picks
    return v[["player_id", "season", "week", "value", "trades_window", "trades_recent", "uncertainty"]]


def window_inputs(P: np.ndarray, G: np.ndarray, val: dict, w: int, k: int) -> dict:
    """The window's points and the form known at its start. ``P``/``G`` are
    weekly points and games played (index 0 = week 1), ``val`` week -> value."""
    win = slice(w - 1, w - 1 + k)
    std = slice(0, w - 1)
    rec = slice(max(0, w - 1 - RECENT), w - 1)
    std_g, rec_g, win_g = G[std].sum(), G[rec].sum(), G[win].sum()
    back = max(1, w - RECENT)
    return {
        "win_points": P[win].sum(), "win_games": win_g,
        "win_ppg": P[win].sum() / win_g if win_g else 0.0,
        "std_ppg": P[std].sum() / std_g if std_g else 0.0, "std_games": std_g,
        "recent_ppg": P[rec].sum() / rec_g if rec_g else 0.0,
        "momentum": float(np.log(val[w] / val[back])) if (val.get(back) or 0) > 0 else 0.0,
    }


def build_windows(pts: pd.DataFrame, values: pd.DataFrame, k: int, min_trades: int, last_value_week: int | None) -> pd.DataFrame:
    """One row per (player, season, start week) with a value at both ends."""
    wide_pts = pts.pivot_table(index=["player_id", "season"], columns="week", values="points", aggfunc="sum").fillna(0.0)
    wide_play = pts.pivot_table(index=["player_id", "season"], columns="week", values="played", aggfunc="sum").fillna(0)
    info = pts.drop_duplicates(["player_id", "season"]).set_index(["player_id", "season"])[["name", "position", "age", "prior_points"]]
    vals = values.set_index(["player_id", "season", "week"])
    rows = []
    for (pid, season), row_pts in wide_pts.iterrows():
        n = regular_weeks(season)
        last = n + 1 if last_value_week is None else min(n + 1, last_value_week)
        P = np.array([row_pts.get(w, 0.0) for w in range(1, n + 1)])
        G = np.array([wide_play.loc[(pid, season)].get(w, 0) for w in range(1, n + 1)])
        try:
            v = vals.loc[(pid, season)]
        except KeyError:
            continue
        val = v["value"].to_dict()
        trades = v["trades_window"].to_dict()
        recent = v["trades_recent"].to_dict() if "trades_recent" in v else {}
        unc = v["uncertainty"].to_dict() if "uncertainty" in v else {}
        meta = info.loc[(pid, season)]
        for w in range(1, n + 2 - k):
            end = w + k
            if end > last or val.get(w, 0) <= 0 or val.get(end, 0) <= 0:
                continue
            if min_trades and (trades.get(w, 0) < min_trades or trades.get(end, 0) < min_trades):
                continue
            rows.append({
                "player_id": pid, "name": meta["name"], "season": season, "week": w, "k": k,
                "position": meta["position"], "age": meta["age"] + (w - 1) / 52.0,
                "start_value": val[w], "end_value": val[end],
                "start_trades": trades.get(w), "end_trades": trades.get(end),
                "end_trades_recent": recent.get(end), "start_unc": unc.get(w), "end_unc": unc.get(end),
                **window_inputs(P, G, val, w, k),
                "prior_points": meta["prior_points"],
            })
    df = pd.DataFrame(rows)
    df["log_start"] = np.log(df["start_value"])
    df["position_code"] = df["position"].map(POSITIONS.index).astype(float)
    df["log_change"] = np.log(df["end_value"] / df["start_value"])
    return df


def build_season(pts: pd.DataFrame, values: pd.DataFrame, min_trades: int, last_value_week: int | None) -> pd.DataFrame:
    """Season labels from the same weekly values: week 1 to the last value week."""
    seasons = pts.drop_duplicates(["player_id", "season"])
    vals = values.set_index(["player_id", "season", "week"])
    rows = []
    for r in seasons.itertuples():
        n = regular_weeks(r.season)
        end_week = n + 1 if last_value_week is None else min(n + 1, last_value_week)
        try:
            a, b = vals.loc[(r.player_id, r.season, 1)], vals.loc[(r.player_id, r.season, end_week)]
        except KeyError:
            continue
        if a["value"] <= 0 or b["value"] <= 0:
            continue
        if min_trades and (a["trades_window"] < min_trades or b["trades_window"] < min_trades):
            continue
        rows.append({"player_id": r.player_id, "season": r.season, "position": r.position, "age": r.age,
                     "points": r.season_points, "games": r.season_games, "prior_points": r.prior_points,
                     "start_value": a["value"], "end_value": b["value"]})
    df = pd.DataFrame(rows)
    df["log_start"] = np.log(df["start_value"])
    df["position_code"] = df["position"].map(POSITIONS.index).astype(float)
    df["ppg"] = np.where(df["games"] > 0, df["points"] / df["games"].clip(lower=1), 0.0)
    return df


# ---------------------------------------------------------------- models


def fit(df: pd.DataFrame, features: list[str], rising: set[str] = RISING) -> HistGradientBoostingRegressor:
    est = HistGradientBoostingRegressor(monotonic_cst=[1 if f in rising else 0 for f in features], **HGB_PARAMS)
    return est.fit(df[features].to_numpy(float), np.log(df["end_value"] / df["start_value"]))


def predict(est, df: pd.DataFrame, features: list[str]) -> np.ndarray:
    return df["start_value"].to_numpy(float) * np.exp(est.predict(df[features].to_numpy(float)))


def season_scaled(season_est, df: pd.DataFrame) -> np.ndarray:
    """The season model asked about a window: the window's pace over a full
    season, its predicted log change scaled down to the window's share of it."""
    n = df["season"].map(regular_weeks).to_numpy(float)
    frac = df["k"].to_numpy(float) / n
    games = df["win_games"].to_numpy(float) / frac
    X = pd.DataFrame({
        "log_start": df["log_start"], "age": df["age"], "points": df["win_points"] / frac, "games": games,
        "ppg": df["win_ppg"], "prior_points": df["prior_points"], "position_code": df["position_code"],
    })[SEASON_FEATURES].to_numpy(float)
    return df["start_value"].to_numpy(float) * np.exp(frac * season_est.predict(X))


def mae(a, b) -> float:
    return float(np.mean(np.abs(np.asarray(a, float) - np.asarray(b, float))))


def score(g: pd.DataFrame, col: str) -> dict:
    base = mae(g["end_value"], g["start_value"])
    m = mae(g["end_value"], g[col])
    return {"mae": round(m, 1), "skill": round(1 - m / base, 4) if base else None}


# ---------------------------------------------------------------- noise


def reversal(df: pd.DataFrame) -> dict:
    """Consecutive non-overlapping windows of one player-season: the
    covariance of the two drift-removed log moves. Independent measurement
    error at the shared date makes it negative (by that error's variance);
    lagging smoothing and real momentum make it positive."""
    d = df.copy()
    d["rel"] = d["log_change"] - d.groupby(["season", "week"])["log_change"].transform("median")
    k = int(d["k"].iloc[0])
    nxt = d[["player_id", "season", "week", "rel"]].assign(week=d["week"] - k).rename(columns={"rel": "rel_next"})
    pair = d.merge(nxt, on=["player_id", "season", "week"])
    if len(pair) < 30:
        return {"pairs": int(len(pair))}
    c = float(np.cov(pair["rel"], pair["rel_next"])[0, 1])
    v = float(pair["rel"].var())
    return {"pairs": int(len(pair)), "var_move": round(v, 5), "cov_next": round(c, 5),
            "corr_next": round(c / v, 3), "noise_share_lower_bound": round(max(0.0, -2 * c / v), 3)}


# ---------------------------------------------------------------- evaluate


def evaluate(source: str, values_path: Path, min_trades: int, train_from: int, ks=(3, 6)) -> dict:
    pts, ktc = read_zip()
    if source == "bc":
        values, last_week = read_bc_values(values_path), None
    else:
        values, last_week, min_trades = ktc, 18, 0
    seasons = build_season(pts, values, min_trades, last_week)
    report = {"source": source, "min_trades": min_trades, "train_from": train_from,
              "selection": list(SELECTION), "check": CHECK, "windows": {}}
    folds = list(SELECTION) + [CHECK]

    season_rows = []
    season_models = {}
    for y in folds:
        tr = seasons[(seasons["season"] >= train_from) & (seasons["season"] < y)]
        te = seasons[seasons["season"] == y].copy()
        est = fit(tr, SEASON_FEATURES, {"points", "ppg"})
        season_models[y] = est
        te["pred"] = predict(est, te, SEASON_FEATURES)
        season_rows.append(te)
    st = pd.concat(season_rows)
    report["season_model"] = {
        str(y): {"n": int((st["season"] == y).sum()), **score(st[st["season"] == y], "pred"),
                 "no_change_mae": round(mae(st.loc[st["season"] == y, "end_value"], st.loc[st["season"] == y, "start_value"]), 1),
                 "rel_mae": round(float(np.mean(np.abs(st.loc[st["season"] == y, "pred"] - st.loc[st["season"] == y, "end_value"]) / st.loc[st["season"] == y, "start_value"])), 4)}
        for y in folds
    }
    report["season_model"]["selection"] = {**score(st[st["season"].isin(SELECTION)], "pred"),
                                           "n": int(st["season"].isin(SELECTION).sum())}

    for k in ks:
        df = build_windows(pts, values, k, min_trades, last_week)
        out = []
        for y in folds:
            tr = df[(df["season"] >= train_from) & (df["season"] < y)]
            te = df[df["season"] == y].copy()
            drift = float(tr["log_change"].median())
            te["pred_drift"] = te["start_value"] * np.exp(drift)
            for name, feats in FEATURES.items():
                te[f"pred_{name}"] = predict(fit(tr, feats), te, feats)
            te["pred_season_scaled"] = season_scaled(season_models[y], te)
            out.append(te)
        res = pd.concat(out)
        cols = ["pred_drift", "pred_season_scaled"] + [f"pred_{n}" for n in FEATURES]

        def block(g: pd.DataFrame) -> dict:
            b = {"n": int(len(g)), "player_seasons": int(g[["player_id", "season"]].drop_duplicates().shape[0]),
                 "no_change_mae": round(mae(g["end_value"], g["start_value"]), 1),
                 "median_abs_move": round(float(np.median(np.abs(g["log_change"]))), 4)}
            for c in cols:
                b[c.removeprefix("pred_")] = score(g, c)
            b["full_rel_mae"] = round(float(np.mean(np.abs(g["pred_full"] - g["end_value"]) / g["start_value"])), 4)
            b["no_change_rel_mae"] = round(float(np.mean(np.abs(g["start_value"] - g["end_value"]) / g["start_value"])), 4)
            return b

        w = {"by_season": {str(y): block(res[res["season"] == y]) for y in folds},
             "selection": block(res[res["season"].isin(SELECTION)]),
             "check": block(res[res["season"] == CHECK])}
        w["by_position_all_folds"] = {p: block(g) for p, g in res.groupby("position")}
        w["by_start_week_all_folds"] = {
            label: block(g) for label, g in res.groupby(pd.cut(res["week"], [0, 4, 8, 12, 19], labels=["1-4", "5-8", "9-12", "13+"]), observed=True)
        }
        # Late season on its own: by the week of the window's last game.
        last_game = res["week"] + res["k"] - 1
        w["by_last_game_week_all_folds"] = {
            str(int(lg)): {**block(g), **({"end_trades_recent_median": float(g["end_trades_recent"].median())} if source == "bc" else {})}
            for lg, g in res.groupby(last_game)
        }
        late = res[res["week"] >= 14]
        w["late_season"] = {
            "rule": "every game of the window in week 14 or later",
            "all_folds": block(late) if len(late) else None,
            "selection": block(late[late["season"].isin(SELECTION)]) if len(late) else None,
            "check": block(late[late["season"] == CHECK]) if (late["season"] == CHECK).any() else None,
            "rest_all_folds": block(res[res["week"] < 14]),
        }
        # How much the what-if input moves the answer: 25th vs 75th percentile window points, same player.
        swing = []
        for y in folds:
            tr = df[(df["season"] >= train_from) & (df["season"] < y)]
            est = fit(tr, FEATURES["full"])
            te = res[res["season"] == y]
            lo, hi = te.copy(), te.copy()
            for pos in POSITIONS:
                m = te["position"] == pos
                if not m.any():
                    continue
                q25, q75 = tr.loc[tr["position"] == pos, "win_ppg"].quantile([0.25, 0.75])
                lo.loc[m, "win_ppg"], hi.loc[m, "win_ppg"] = q25, q75
            for frame in (lo, hi):
                frame["win_games"] = np.maximum(frame["win_games"], 1)
                frame["win_points"] = frame["win_ppg"] * frame["win_games"]
            swing.append(np.log(predict(est, hi, FEATURES["full"]) / predict(est, lo, FEATURES["full"])))
        swing = np.concatenate(swing)
        w["points_swing"] = {"median_log_q75_vs_q25": round(float(np.median(swing)), 4)}
        if source == "bc":
            bins = pd.cut(res["end_trades_recent"], [-1, 4, 14, 39, 10**9], labels=["0-4", "5-14", "15-39", "40+"])
            w["by_recent_trades_at_end"] = {str(b): {**block(g), "reversal": reversal(g)} for b, g in res.groupby(bins, observed=True)}
            sbins = pd.cut(res["start_trades"], [4, 19, 49, 149, 10**9], labels=["5-19", "20-49", "50-149", "150+"])
            w["by_trades_window_at_start"] = {str(b): {**block(g), "reversal": reversal(g)} for b, g in res.groupby(sbins, observed=True)}
            w["trades_behind_labels"] = {
                "start_trades_window_median": float(res["start_trades"].median()),
                "end_trades_recent_median": float(res["end_trades_recent"].median()),
                "end_trades_recent_share_under_5": round(float((res["end_trades_recent"] < 5).mean()), 3),
            }
        w["reversal_all"] = reversal(res)
        report["windows"][str(k)] = w
    return report


def main(argv=None) -> None:
    ap = argparse.ArgumentParser(prog="python -m predictor_v5.short_window")
    ap.add_argument("command", choices=["evaluate"])
    ap.add_argument("--source", choices=["bc", "ktc"], default="bc")
    ap.add_argument("--values", type=Path, default=DEFAULT_VALUES)
    ap.add_argument("--min-trades", type=int, default=5)
    ap.add_argument("--train-from", type=int, default=None)
    ap.add_argument("--out", type=Path, default=None)
    a = ap.parse_args(argv)
    train_from = a.train_from or (2020 if a.source == "bc" else 2021)
    rep = evaluate(a.source, a.values, a.min_trades, train_from)
    out = a.out or ARTIFACTS / f"short_window_{a.source}_eval.json"
    out.write_text(json.dumps(rep, indent=2, allow_nan=False) + "\n")
    print(json.dumps({k: rep[k] for k in ("source", "season_model")}, indent=1))
    for k, w in rep["windows"].items():
        print(k, "selection", json.dumps(w["selection"]))
        print(k, "check", json.dumps(w["check"]))
    print("wrote", out)


if __name__ == "__main__":
    main()
