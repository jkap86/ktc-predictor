"""Analyze whether comp avg outcomes are predictive beyond the model."""

import json
import sys
import zipfile
from pathlib import Path

import numpy as np
import pandas as pd

# Add backend to path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from app.services.comps import get_comps_index
from app.services.model_registry import get_registry


def load_season_data():
    """Load training data and build per-season records."""
    data_path = Path(__file__).resolve().parent.parent / "data"
    zip_path = data_path / "training-data.zip"
    with zipfile.ZipFile(zip_path) as zf:
        with zf.open("training-data.json") as f:
            raw = json.load(f)

    records = []
    for player in raw["players"]:
        pid = player["player_id"]
        pos = player["position"]
        seasons = sorted(player.get("seasons", []), key=lambda s: s["year"])

        for i, s in enumerate(seasons):
            start_ktc = s.get("start_ktc")
            end_ktc = s.get("end_ktc")
            gp = s.get("games_played", 0) or 0
            fp = s.get("fantasy_points", 0) or 0

            if not start_ktc or start_ktc <= 0 or not end_ktc or end_ktc <= 0:
                continue
            if gp < 1:
                continue

            ppg = fp / gp
            age = s.get("age") or 25
            years_exp = s.get("years_exp") or 0

            # Prior season for prior features
            prior = None
            for j in range(i - 1, -1, -1):
                if (seasons[j].get("games_played", 0) or 0) >= 4:
                    prior = seasons[j]
                    break

            prior_ppg = 0.0
            if prior and (prior.get("games_played", 0) or 0) > 0:
                prior_ppg = (prior.get("fantasy_points", 0) or 0) / prior["games_played"]

            ktc_yoy_pct = 0.0
            if prior and (prior.get("end_ktc") or 0) > 0:
                ktc_yoy_pct = (start_ktc - prior["end_ktc"]) / prior["end_ktc"]

            records.append({
                "player_id": pid,
                "name": player["name"],
                "position": pos,
                "year": s["year"],
                "age": age,
                "years_exp": years_exp,
                "games_played": gp,
                "ppg": round(ppg, 2),
                "start_ktc": start_ktc,
                "end_ktc": end_ktc,
                "delta_ktc": end_ktc - start_ktc,
                "log_ratio": np.log(max(end_ktc, 1) / start_ktc),
                "prior_ppg": prior_ppg,
                "ktc_yoy_pct": ktc_yoy_pct,
                "start_position_rank": s.get("start_position_rank"),
                # Pass-through for comp matching
                "boom_rate": s.get("boom_rate"),
                "bust_rate": s.get("bust_rate"),
                "snap_pct": s.get("snap_pct"),
                "ktc_30d_trend": s.get("ktc_30d_trend"),
                "ktc_volatility": s.get("ktc_volatility"),
                "momentum_ratio": s.get("momentum_ratio"),
                "max_games_missed_streak": s.get("max_games_missed_streak"),
                "draft_pick": s.get("draft_pick"),
            })

    return pd.DataFrame(records)


def run_analysis():
    print("Loading data...")
    df = load_season_data()
    print(f"  {len(df)} player-seasons loaded")

    print("Loading model and comps index...")
    registry = get_registry()
    iteration = registry.get()
    comps_index = get_comps_index(iteration.id)

    results = []
    positions = ["QB", "RB", "WR", "TE"]

    for pos in positions:
        pos_df = df[df["position"] == pos].copy()
        print(f"\n{'='*60}")
        print(f"  {pos}: {len(pos_df)} seasons")

        for _, row in pos_df.iterrows():
            try:
                comps = comps_index.find_comps(
                    position=pos,
                    start_ktc=row["start_ktc"],
                    ppg=row["ppg"],
                    age=float(row["age"]),
                    games_played=row["games_played"],
                    k=10,
                    exclude_player_id=row["player_id"],
                    weeks_missed=0,
                    draft_pick=row.get("draft_pick"),
                    start_position_rank=row.get("start_position_rank"),
                    years_exp=row.get("years_exp"),
                    ktc_yoy_pct=row.get("ktc_yoy_pct", 0),
                    prior_ppg=row.get("prior_ppg", 0),
                    boom_rate=row.get("boom_rate"),
                    bust_rate=row.get("bust_rate"),
                    snap_pct=row.get("snap_pct"),
                    ktc_30d_trend=row.get("ktc_30d_trend"),
                    ktc_volatility=row.get("ktc_volatility"),
                    momentum_ratio=row.get("momentum_ratio"),
                    max_games_missed_streak=row.get("max_games_missed_streak"),
                )
            except Exception:
                continue

            if len(comps) < 3:
                continue

            comp_deltas = [c["delta_ktc"] for c in comps]
            comp_avg_delta = np.mean(comp_deltas)
            comp_median_delta = np.median(comp_deltas)
            comp_std = np.std(comp_deltas)
            comp_risers_pct = sum(1 for d in comp_deltas if d > 0) / len(comp_deltas)
            avg_similarity = np.mean([c["similarity"] for c in comps])

            results.append({
                "player_id": row["player_id"],
                "name": row["name"],
                "position": pos,
                "year": row["year"],
                "start_ktc": row["start_ktc"],
                "actual_delta": row["delta_ktc"],
                "comp_avg_delta": comp_avg_delta,
                "comp_median_delta": comp_median_delta,
                "comp_std": comp_std,
                "comp_risers_pct": comp_risers_pct,
                "avg_similarity": avg_similarity,
                "num_comps": len(comps),
            })

    res = pd.DataFrame(results)
    print(f"\n{'='*60}")
    print(f"Total analyzed: {len(res)} player-seasons\n")

    # ── Analysis 1: Correlation of comp avg delta with actual delta ──
    print("--- CORRELATION: Comp Avg Delta vs Actual Delta ---")
    for pos in positions:
        pf = res[res["position"] == pos]
        if len(pf) < 10:
            continue
        corr = pf["comp_avg_delta"].corr(pf["actual_delta"])
        print(f"  {pos}: r = {corr:.3f}  (n={len(pf)})")
    all_corr = res["comp_avg_delta"].corr(res["actual_delta"])
    print(f"  ALL: r = {all_corr:.3f}  (n={len(res)})")

    # ── Analysis 2: Comp avg delta as a predictor — MAE ──
    print("\n--- MAE: Comp Avg Delta as Predictor ---")
    print(f"  {'Pos':<5} {'Comp MAE':>10} {'No-Change MAE':>15} {'Comp Better?':>14}")
    for pos in positions:
        pf = res[res["position"] == pos]
        if len(pf) < 10:
            continue
        comp_mae = np.mean(np.abs(pf["actual_delta"] - pf["comp_avg_delta"]))
        naive_mae = np.mean(np.abs(pf["actual_delta"]))  # predict delta=0
        print(f"  {pos:<5} {comp_mae:>10.1f} {naive_mae:>15.1f} {'YES' if comp_mae < naive_mae else 'no':>14}")

    # ── Analysis 3: Residual analysis — does comp info help after model? ──
    # Use comp_avg_delta as a correction: blend = model + alpha*(comp_avg - model)
    print("\n--- BLEND ANALYSIS: Model + Comp Avg ---")
    print("  If comp avg adds value, blending should beat model alone.")
    print(f"  Testing alpha values (0 = model only, 1 = comp only)\n")

    # We don't have model predictions here, so use comp_avg vs actual directly
    # and measure if comp direction agrees with actual
    print("--- DIRECTION AGREEMENT: Did comps predict direction? ---")
    for pos in positions:
        pf = res[res["position"] == pos]
        if len(pf) < 10:
            continue
        same_dir = ((pf["comp_avg_delta"] > 0) == (pf["actual_delta"] > 0)).mean()
        print(f"  {pos}: {same_dir:.1%} direction match")

    # ── Analysis 4: Comp variance as confidence signal ──
    print("\n--- COMP VARIANCE vs ACTUAL ERROR ---")
    print("  If comp std correlates with actual deviation, it's useful for confidence bands.")
    for pos in positions:
        pf = res[res["position"] == pos]
        if len(pf) < 10:
            continue
        # Does high comp variance predict large actual deviations from comp avg?
        pf = pf.copy()
        pf["comp_error"] = np.abs(pf["actual_delta"] - pf["comp_avg_delta"])
        corr = pf["comp_std"].corr(pf["comp_error"])
        print(f"  {pos}: r = {corr:.3f} (comp_std vs |actual - comp_avg|)")

    # ── Analysis 5: Comp risers % as signal ──
    print("\n--- COMP RISERS % vs ACTUAL DIRECTION ---")
    for pos in positions:
        pf = res[res["position"] == pos]
        if len(pf) < 10:
            continue
        # Split into high/low comp risers %
        high = pf[pf["comp_risers_pct"] >= 0.6]
        low = pf[pf["comp_risers_pct"] <= 0.4]
        high_actual_rise = (high["actual_delta"] > 0).mean() if len(high) > 0 else 0
        low_actual_rise = (low["actual_delta"] > 0).mean() if len(low) > 0 else 0
        print(f"  {pos}: >=60% comps rose -> {high_actual_rise:.1%} actually rose (n={len(high)})")
        print(f"  {pos}: <=40% comps rose -> {low_actual_rise:.1%} actually rose (n={len(low)})")

    print("\n--- SIMILARITY ANALYSIS ---")
    print("  Are higher-similarity comps more predictive?")
    for pos in positions:
        pf = res[res["position"] == pos]
        if len(pf) < 10:
            continue
        pf = pf.copy()
        pf["comp_error"] = np.abs(pf["actual_delta"] - pf["comp_avg_delta"])
        corr = pf["avg_similarity"].corr(pf["comp_error"])
        print(f"  {pos}: r = {corr:.3f} (similarity vs error — negative = better)")


if __name__ == "__main__":
    run_analysis()
