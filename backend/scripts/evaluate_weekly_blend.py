"""Evaluate weekly blend approach for Experiment A.

Compares EOS-only predictions vs EOS+weekly blended predictions
to measure improvement in elite tier accuracy.
"""

import json
import zipfile
from pathlib import Path

import numpy as np
import pandas as pd

from app.services.data_loader import DataLoader
from app.services.eos_model_service import EosModelService
from app.services.transition_model_service import TransitionModelService


def evaluate_blend(data_path: str = "data/training-data.zip") -> dict:
    """Evaluate weekly blend on all players with weekly data."""

    # Load data
    with zipfile.ZipFile(data_path, "r") as zf:
        with zf.open("training-data.json") as f:
            data = json.load(f)

    # Initialize services
    eos_service = EosModelService()
    eos_service.initialize()

    transition_service = TransitionModelService()
    transition_service.initialize()

    # Create data loader with extracted data
    data_loader = DataLoader(Path("data/training-data.json"))

    results = []

    for player in data["players"]:
        player_id = player["player_id"]
        position = player["position"]

        for season in player.get("seasons", []):
            # Need both start and end KTC
            start_ktc = season.get("start_ktc", 0)
            end_ktc = season.get("end_ktc", 0)

            if not start_ktc or not end_ktc or start_ktc <= 0 or end_ktc <= 0:
                continue
            if start_ktc >= 9999 or end_ktc >= 9999:
                continue

            weekly_stats = season.get("weekly_stats", [])
            weekly_ktc = season.get("weekly_ktc", [])

            # Need weekly data for transition model
            if not weekly_stats or not weekly_ktc:
                continue

            games_played = sum(ws.get("games_played", 0) for ws in weekly_stats)
            total_fp = sum(ws.get("fantasy_points", 0) or 0 for ws in weekly_stats)
            ppg = max(0.0, total_fp / games_played if games_played > 0 else 0)
            age = season.get("age")

            # Get EOS prediction
            eos_result = eos_service.predict_from_inputs(
                position=position,
                start_ktc=start_ktc,
                games_played=games_played,
                ppg=ppg,
                age=float(age) if age else None,
            )
            eos_pred = eos_result["predicted_end_ktc"]

            # Get weekly rollout prediction
            model = transition_service.bundle["models"].get(position)
            clip_bounds = transition_service.bundle["clip_bounds"].get(position, (-0.5, 0.5))

            if model is None:
                continue

            # Rollout from start to get weekly prediction
            from ktc_model.predict_transition import predict_end_ktc_via_rollout

            weekly_result = predict_end_ktc_via_rollout(
                model=model,
                clip_bounds=clip_bounds,
                start_ktc=start_ktc,
                weekly_stats=weekly_stats,
                age=age,
                position=position,
            )
            weekly_pred = weekly_result["end_ktc"]

            # Blend based on games played
            if games_played <= 4:
                weekly_weight = 0.4
            elif games_played <= 8:
                weekly_weight = 0.3
            elif games_played <= 12:
                weekly_weight = 0.2
            else:
                weekly_weight = 0.1

            blended_pred = (1 - weekly_weight) * eos_pred + weekly_weight * weekly_pred
            blended_pred = max(1.0, min(9999.0, blended_pred))

            # Compute errors
            eos_error = eos_pred - end_ktc
            weekly_error = weekly_pred - end_ktc
            blended_error = blended_pred - end_ktc

            actual_delta = end_ktc - start_ktc
            is_riser = actual_delta > 0

            # Tier
            if start_ktc < 2000:
                tier = "0-2k"
            elif start_ktc < 4000:
                tier = "2k-4k"
            elif start_ktc < 6000:
                tier = "4k-6k"
            else:
                tier = "6k+"

            results.append({
                "player_id": player_id,
                "position": position,
                "year": season["year"],
                "start_ktc": start_ktc,
                "end_ktc": end_ktc,
                "games_played": games_played,
                "tier": tier,
                "is_riser": is_riser,
                "eos_pred": eos_pred,
                "weekly_pred": weekly_pred,
                "blended_pred": blended_pred,
                "eos_error": eos_error,
                "weekly_error": weekly_error,
                "blended_error": blended_error,
                "blend_weight": weekly_weight,
            })

    df = pd.DataFrame(results)

    # Compute metrics
    print("\n" + "=" * 70)
    print("EXPERIMENT A: WEEKLY BLEND EVALUATION")
    print("=" * 70)

    tiers = ["0-2k", "2k-4k", "4k-6k", "6k+"]

    for pos in ["QB", "RB", "WR", "TE"]:
        pos_df = df[df["position"] == pos]
        if len(pos_df) == 0:
            continue

        eos_mae = np.mean(np.abs(pos_df["eos_error"]))
        weekly_mae = np.mean(np.abs(pos_df["weekly_error"]))
        blended_mae = np.mean(np.abs(pos_df["blended_error"]))

        print(f"\n{pos} (n={len(pos_df)}):")
        print(f"  Overall MAE:  EOS={eos_mae:.0f}  Weekly={weekly_mae:.0f}  Blended={blended_mae:.0f}")

        # Tier-level analysis
        print(f"\n  Tier Analysis:")
        for tier in tiers:
            tier_df = pos_df[pos_df["tier"] == tier]
            if len(tier_df) < 10:
                continue

            tier_eos_mae = np.mean(np.abs(tier_df["eos_error"]))
            tier_blended_mae = np.mean(np.abs(tier_df["blended_error"]))
            tier_eos_bias = np.mean(tier_df["eos_error"])
            tier_blended_bias = np.mean(tier_df["blended_error"])

            # Riser/faller analysis for elite tiers
            if tier in ["4k-6k", "6k+"]:
                risers = tier_df[tier_df["is_riser"]]
                fallers = tier_df[~tier_df["is_riser"]]

                eos_riser_bias = np.mean(risers["eos_error"]) if len(risers) > 5 else None
                blended_riser_bias = np.mean(risers["blended_error"]) if len(risers) > 5 else None

                riser_str = ""
                if eos_riser_bias is not None:
                    improvement = eos_riser_bias - blended_riser_bias
                    riser_str = f"  Riser: EOS={eos_riser_bias:+.0f} -> Blend={blended_riser_bias:+.0f} ({improvement:+.0f})"

                print(f"    {tier}: n={len(tier_df):3d}  MAE: EOS={tier_eos_mae:.0f} -> Blend={tier_blended_mae:.0f}  "
                      f"Bias: EOS={tier_eos_bias:+.0f} -> Blend={tier_blended_bias:+.0f}{riser_str}")
            else:
                print(f"    {tier}: n={len(tier_df):3d}  MAE: EOS={tier_eos_mae:.0f} -> Blend={tier_blended_mae:.0f}  "
                      f"Bias: EOS={tier_eos_bias:+.0f} -> Blend={tier_blended_bias:+.0f}")

    # Summary comparison
    print("\n" + "=" * 70)
    print("SUMMARY: ELITE TIER (6k+) RISER BIAS COMPARISON")
    print("=" * 70)

    baseline = {
        "QB": {"riser_bias": 705.5},
        "RB": {"riser_bias": 1164.2},
        "WR": {"riser_bias": -25.3},
        "TE": {"riser_bias": -312.7},
    }

    for pos in ["QB", "RB", "WR", "TE"]:
        elite_risers = df[(df["position"] == pos) & (df["tier"] == "6k+") & (df["is_riser"])]
        if len(elite_risers) < 5:
            continue

        eos_riser_bias = np.mean(elite_risers["eos_error"])
        blended_riser_bias = np.mean(elite_risers["blended_error"])
        baseline_bias = baseline[pos]["riser_bias"]

        # Improvement from baseline
        eos_vs_baseline = abs(eos_riser_bias) - abs(baseline_bias)
        blend_vs_baseline = abs(blended_riser_bias) - abs(baseline_bias)

        print(f"{pos} 6k+ Risers (n={len(elite_risers)}):")
        print(f"  Baseline bias: {baseline_bias:+.0f}")
        print(f"  EOS-only bias: {eos_riser_bias:+.0f}")
        print(f"  Blended bias:  {blended_riser_bias:+.0f}")
        improvement_pct = (1 - abs(blended_riser_bias) / abs(eos_riser_bias)) * 100 if eos_riser_bias != 0 else 0
        print(f"  Improvement:   {improvement_pct:+.1f}%\n")

    return {"results": results}


if __name__ == "__main__":
    evaluate_blend()
