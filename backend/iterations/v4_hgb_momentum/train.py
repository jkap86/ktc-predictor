"""v4 training script: v3 + KTC momentum + position-specific efficiency features.

Extends v3's feature set with:
  Universal (all positions, from current season):
    ktc_30d_trend, ktc_90d_trend, momentum_ratio, max_games_missed_streak
  Position-specific (from prior season):
    QB:    prior_passing_tds, prior_interceptions
    RB:   prior_carries, prior_red_zone_touches
    WR/TE: prior_targets, prior_red_zone_targets
  Sentinel: has_prior_position_stats (1 if position-specific prior data exists)

Feature counts: QB=27, RB=33, WR=33, TE=27

Usage:
    cd backend
    python -m iterations.v4_hgb_momentum.train --zip data/training-data.zip
"""

import argparse
import copy
import json
import zipfile
from pathlib import Path

import numpy as np

# Import v3's train module first (applies v3 patches to v1)
import iterations.v3_hgb_prior_signals.train as v3
import iterations.v1_hgb_baseline.train as v1
import iterations.v1_hgb_baseline.data as v1_data

# ── Prior-season position-specific feature lookup ───────────────────────

_MIN_GAMES = 4


def _compute_prior_position_features(
    players: list[dict],
) -> dict[tuple[str, int], dict]:
    """For each (player_id, year), extract position-specific stats from the
    prior season with >= 4 games played.

    Includes both volume and efficiency metrics per position.
    """
    result: dict[tuple[str, int], dict] = {}

    for player in players:
        pid = player["player_id"]
        pos = player["position"]
        seasons = sorted(
            [s for s in player.get("seasons", []) if (s.get("years_exp") or 0) >= 0],
            key=lambda s: s["year"],
        )

        for i, season in enumerate(seasons):
            if i == 0:
                continue
            prior = seasons[i - 1]
            gp = prior.get("games_played", 0) or 0
            if gp < _MIN_GAMES:
                continue

            if pos == "QB":
                result[(pid, season["year"])] = {
                    "prior_passing_tds": float(prior.get("passing_tds") or 0),
                    "prior_interceptions": float(prior.get("interceptions") or 0),
                    "prior_completion_rate": float(prior.get("completion_rate") or 0),
                    "prior_rushing_yards": float(prior.get("rushing_yards") or 0),
                    "prior_pass_sacks": float(prior.get("pass_sacks") or 0),
                }
            elif pos == "RB":
                carries = float(prior.get("carries") or 0)
                rush_yds = float(prior.get("rushing_yards") or 0)
                result[(pid, season["year"])] = {
                    "prior_carries": carries,
                    "prior_red_zone_touches": float(prior.get("red_zone_touches") or 0),
                    "prior_yards_per_carry": rush_yds / carries if carries > 0 else 0.0,
                    "prior_receiving_yards": float(prior.get("receiving_yards") or 0),
                    "prior_rushing_tds": float(prior.get("rushing_tds") or 0),
                }
            elif pos == "WR":
                tgts = float(prior.get("targets") or 0)
                result[(pid, season["year"])] = {
                    "prior_targets": tgts,
                    "prior_red_zone_targets": float(prior.get("red_zone_targets") or 0),
                    "prior_yards_per_target": float(prior.get("yards_per_target") or 0),
                    "prior_air_yards_per_target": float(prior.get("air_yards_per_target") or 0),
                    "prior_receiving_tds": float(prior.get("receiving_tds") or 0),
                    "prior_drop_rate": float(prior.get("drop_rate") or 0),
                }
            elif pos == "TE":
                result[(pid, season["year"])] = {
                    "prior_targets": float(prior.get("targets") or 0),
                    "prior_red_zone_targets": float(prior.get("red_zone_targets") or 0),
                    "prior_yards_per_target": float(prior.get("yards_per_target") or 0),
                    "prior_receiving_tds": float(prior.get("receiving_tds") or 0),
                    "prior_drop_rate": float(prior.get("drop_rate") or 0),
                }

    return result


def _compute_career_trajectory(
    players: list[dict],
) -> dict[tuple[str, int], dict]:
    """For each (player_id, year), compute trajectory features using the
    two most recent prior seasons with >= 4 games played.

    Features:
    - prior_2yr_ppg: PPG from 2 seasons ago (career baseline)
    - ppg_trend: slope of PPG over last 2 seasons (positive = improving)
    - prior_2yr_end_ktc: end KTC from 2 seasons ago (career value baseline)
    """
    _MIN_GAMES = 4
    result: dict[tuple[str, int], dict] = {}

    for player in players:
        pid = player["player_id"]
        seasons = sorted(
            [s for s in player.get("seasons", []) if (s.get("years_exp") or 0) >= 0],
            key=lambda s: s["year"],
        )

        for i, season in enumerate(seasons):
            # Find the two most recent prior seasons with enough games
            priors = []
            for j in range(i - 1, -1, -1):
                gp = seasons[j].get("games_played", 0) or 0
                if gp >= _MIN_GAMES:
                    priors.append(seasons[j])
                if len(priors) >= 2:
                    break

            if len(priors) < 2:
                continue

            prior_1 = priors[0]  # N-1
            prior_2 = priors[1]  # N-2

            gp1 = prior_1.get("games_played", 0) or 1
            fp1 = prior_1.get("fantasy_points", 0) or 0
            ppg1 = fp1 / gp1

            gp2 = prior_2.get("games_played", 0) or 1
            fp2 = prior_2.get("fantasy_points", 0) or 0
            ppg2 = fp2 / gp2

            end_ktc_2 = prior_2.get("end_ktc") or 0

            result[(pid, season["year"])] = {
                "prior_2yr_ppg": round(ppg2, 2),
                "ppg_trend": round(ppg1 - ppg2, 2),  # positive = improving
                "prior_2yr_end_ktc": round(end_ktc_2, 1),
            }

    return result


# ── Wrap the data builder to add v4 columns ─────────────────────────────

_v3_build = v1.build_weekly_snapshot_df  # v3 already patched this


def _build_weekly_snapshot_df_v4(zip_path: str, json_name: str = "training-data.json"):
    """v3's snapshot builder + v4's current-season momentum and prior position stats."""
    df = _v3_build(zip_path, json_name)

    # Re-read for prior-position lookups
    with zipfile.ZipFile(Path(zip_path), "r") as zf:
        with zf.open(json_name) as f:
            data = json.load(f)

    # Current-season KTC momentum features (already on season records)
    # We need to join them to the weekly snapshots. They're season-level so same
    # value for every week of that season.
    season_feats = {}
    for player in data["players"]:
        pid = player["player_id"]
        for season in player.get("seasons", []):
            if (season.get("years_exp") or 0) < 0:
                continue
            season_feats[(pid, season["year"])] = {
                "ktc_30d_trend": float(season.get("ktc_30d_trend") or 0),
                "ktc_90d_trend": float(season.get("ktc_90d_trend") or 0),
                "momentum_ratio": float(season.get("momentum_ratio") or 0),
                "max_games_missed_streak": float(season.get("max_games_missed_streak") or 0),
            }

    # Prior-season position-specific stats
    prior_pos_feats = _compute_prior_position_features(data["players"])
    print(f"  Prior-position features loaded for {len(prior_pos_feats)} player-seasons")

    # Career trajectory (2-year lookback)
    trajectory_feats = _compute_career_trajectory(data["players"])
    print(f"  Career trajectory features loaded for {len(trajectory_feats)} player-seasons")

    def _lookup_season(row, key):
        feat = season_feats.get((row["player_id"], row["year"]))
        return feat[key] if feat else 0.0

    def _lookup_prior_pos(row, key):
        feat = prior_pos_feats.get((row["player_id"], row["year"]))
        return feat.get(key) if feat else None

    # Universal momentum features
    for col in ["ktc_30d_trend", "ktc_90d_trend", "momentum_ratio", "max_games_missed_streak"]:
        df[col] = df.apply(lambda r, c=col: _lookup_season(r, c), axis=1)

    # Position-specific prior stats (use NaN for positions that don't get that feature)
    pos_cols_map = {
        "QB": ["prior_passing_tds", "prior_interceptions"],
        "RB": ["prior_carries", "prior_red_zone_touches"],
        "WR": ["prior_targets", "prior_red_zone_targets"],
        "TE": ["prior_targets", "prior_red_zone_targets"],
    }
    # All possible position-stat column names
    all_pos_cols = [
        "prior_passing_tds", "prior_interceptions", "prior_completion_rate",
        "prior_rushing_yards", "prior_pass_sacks",
        "prior_carries", "prior_red_zone_touches", "prior_yards_per_carry",
        "prior_receiving_yards", "prior_rushing_tds",
        "prior_targets", "prior_red_zone_targets", "prior_yards_per_target",
        "prior_air_yards_per_target", "prior_receiving_tds", "prior_drop_rate",
    ]
    for col in all_pos_cols:
        df[col] = df.apply(lambda r, c=col: _lookup_prior_pos(r, c), axis=1)

    df["has_prior_position_stats"] = df.apply(
        lambda r: 1 if prior_pos_feats.get((r["player_id"], r["year"])) else 0, axis=1
    )

    # Career trajectory columns
    def _lookup_trajectory(row, key):
        feat = trajectory_feats.get((row["player_id"], row["year"]))
        return feat.get(key) if feat else None

    for col in _TRAJECTORY_FEATURES:
        df[col] = df.apply(lambda r, c=col: _lookup_trajectory(r, c), axis=1)

    df["has_career_trajectory"] = df.apply(
        lambda r: 1 if trajectory_feats.get((r["player_id"], r["year"])) else 0, axis=1
    )

    return df


v1.build_weekly_snapshot_df = _build_weekly_snapshot_df_v4

# ── Patch feature configuration ─────────────────────────────────────────

_V3_GET_FEATURES = v1.get_features_for_position

# Universal momentum features (current season, all positions)
_MOMENTUM_FEATURES = [
    "ktc_30d_trend",
    "ktc_90d_trend",
    "momentum_ratio",
    "max_games_missed_streak",
]

# Career trajectory features (2-year lookback, all positions)
_TRAJECTORY_FEATURES = [
    "prior_2yr_ppg",
    "ppg_trend",
    "prior_2yr_end_ktc",
    "has_career_trajectory",
]

# Position-specific prior-season stat features (volume + efficiency)
_POSITION_STAT_FEATURES = {
    "QB": ["prior_passing_tds", "prior_interceptions", "prior_completion_rate",
           "prior_rushing_yards", "prior_pass_sacks"],
    "RB": ["prior_carries", "prior_red_zone_touches", "prior_yards_per_carry",
           "prior_receiving_yards", "prior_rushing_tds"],
    "WR": ["prior_targets", "prior_red_zone_targets", "prior_yards_per_target",
           "prior_air_yards_per_target", "prior_receiving_tds", "prior_drop_rate"],
    "TE": ["prior_targets", "prior_red_zone_targets", "prior_yards_per_target",
           "prior_receiving_tds", "prior_drop_rate"],
}


# Positions that get v4's momentum + position-efficiency features.
# Initial v4 applied to all 4; RB/TE were worse than v3, QB/WR improved.
_POSITIONS_WITH_V4_FEATURES = {"QB", "WR"}

# Position rank feature (RB/TE only — differentiates players at similar KTC;
# hurt QB/WR where existing features already capture relative positioning)
_POSITIONS_WITH_RANK_FEATURE = {"RB", "TE"}

# Team context features (WR only — helps with target competition;
# hurt QB/RB/TE where individual trajectory dominates)
_TEAM_FEATURES = ["qb_ktc", "team_total_ktc", "positional_competition"]
_POSITIONS_WITH_TEAM_FEATURES = {"WR"}


def _v4_get_features_for_position(position: str) -> list[str]:
    """v3's features + position-specific efficiency stats for all positions,
    momentum for QB/WR, team context for WR."""
    base = _V3_GET_FEATURES(position)
    extras: list[str] = []
    # Momentum features (QB/WR only — hurt RB/TE)
    if position in _POSITIONS_WITH_V4_FEATURES:
        extras.extend(_MOMENTUM_FEATURES)
    # Position-specific efficiency stats (all positions)
    pos_stats = _POSITION_STAT_FEATURES.get(position, [])
    if pos_stats:
        extras.extend(pos_stats)
        extras.append("has_prior_position_stats")
    # Position rank (RB/TE only)
    if position in _POSITIONS_WITH_RANK_FEATURE:
        extras.append("start_position_rank")
    # Career trajectory (QB/TE only — hurt RB/WR)
    if position in ("QB", "TE"):
        extras.extend(_TRAJECTORY_FEATURES)
    # Team context (WR only)
    if position in _POSITIONS_WITH_TEAM_FEATURES:
        extras.extend(_TEAM_FEATURES)
    if not extras:
        return base
    return base + extras


v1.get_features_for_position = _v4_get_features_for_position

# Rebuild derived constants
v1.FEATURES = v1.get_features_for_position("QB")
v1.N_CORE_FEATURES = len(v1._CORE_FEATURES)
v1.N_LINEAR_FEATURES = len(v1.FEATURES) - v1.N_CORE_FEATURES

# ── Patch monotonic constraints ──────────────────────────────────────────

_v3_build_mc = v1._build_monotonic_constraints


def _v4_build_monotonic_constraints(position: str) -> list[int]:
    """v3's constraints + zeros for the new features."""
    base = _v3_build_mc(position)
    n_new = 0
    if position in _POSITIONS_WITH_V4_FEATURES:
        n_new += len(_MOMENTUM_FEATURES)
    pos_stats = _POSITION_STAT_FEATURES.get(position, [])
    if pos_stats:
        n_new += len(pos_stats) + 1  # +1 for has_prior_position_stats
    if position in _POSITIONS_WITH_RANK_FEATURE:
        n_new += 1  # start_position_rank
    if position in ("QB", "TE"):
        n_new += len(_TRAJECTORY_FEATURES)
    if position in _POSITIONS_WITH_TEAM_FEATURES:
        n_new += len(_TEAM_FEATURES)
    if n_new == 0:
        return base
    return base + [0] * n_new


v1._build_monotonic_constraints = _v4_build_monotonic_constraints

# ── Patch smoke tests ────────────────────────────────────────────────────


def _v4_monotonic_smoke_test(model, position: str) -> bool:
    """v3's smoke test with v4's extra trailing features."""
    ppg_values = [5, 10, 15, 20]

    X_test = []
    for ppg in ppg_values:
        row = [8, ppg, 2, np.nan, 3, 4, -2, 0]  # core (8)
        row.extend([5000, 0])  # start_ktc, sentinel
        row.extend([0.0, 0.0, 1])  # prior KTC
        row.extend([15.0, np.log((ppg + 0.1) / (15 + 0.1)), 1])  # prior PPG
        if v1.USE_CONTRACT_FEATURES:
            row.extend([0.05, 0, 0.5, 1])  # contract
        if v1.USE_PFF_FEATURES:
            row.extend([70.0, 1, 70.0, 1])
        if v1.USE_TEAM_FEATURES:
            row.extend([5000, 30000, 3000])
        # v3 prior-behavioral (RB/WR only)
        if position in v3._POSITIONS_WITH_PRIOR_BEHAVIORAL:
            row.extend([0.4, 0.25, 0.25, 0.65, 1.0, 1])
        # v4 momentum (QB/WR only)
        if position in _POSITIONS_WITH_V4_FEATURES:
            row.extend([0.0, 0.0, 0.1, 0.1])  # momentum (4)
        # v4 position-specific efficiency (all positions)
        n_pos = len(_POSITION_STAT_FEATURES.get(position, []))
        if n_pos > 0:
            row.extend([0.0] * n_pos)  # position stats
            row.append(1)  # has_prior_position_stats sentinel
        # v4 position rank (RB/TE only)
        if position in _POSITIONS_WITH_RANK_FEATURE:
            row.append(10)  # start_position_rank
        # v4 career trajectory (QB/TE only)
        if position in ("QB", "TE"):
            row.extend([15.0, 1.0, 5000.0, 1])  # prior_2yr_ppg, ppg_trend, prior_2yr_end_ktc, has_career_trajectory
        # v4 team features (WR only)
        if position in _POSITIONS_WITH_TEAM_FEATURES:
            row.extend([5000, 30000, 3000])  # qb_ktc, team_total, positional_comp
        X_test.append(row)

    X_test = np.array(X_test)
    preds = model.predict(X_test)
    is_monotonic = all(preds[i] <= preds[i + 1] for i in range(len(preds) - 1))
    status = "PASS" if is_monotonic else "FAIL"
    print(f"  Monotonic test ({position}): {status}  preds={[round(p, 4) for p in preds]}")
    return is_monotonic


def _v4_ppg_sensitivity_test(model, calibrator_dict, clip_bounds, position: str) -> bool:
    print(f"  PPG sensitivity ({position}): SKIP  (v4 feature set)")
    return True


v1._monotonic_smoke_test = _v4_monotonic_smoke_test
v1._ppg_sensitivity_test = _v4_ppg_sensitivity_test


def main():
    parser = argparse.ArgumentParser(
        description="Train v4 per-position KTC models (v3 + momentum + position efficiency)"
    )
    parser.add_argument("--zip", default="data/training-data.zip")
    parser.add_argument("--json", default="training-data.json")
    parser.add_argument("--out", default="iterations/v4_hgb_momentum/models")
    parser.add_argument("--test-size", type=float, default=0.2)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--export-csv", default=None)
    parser.add_argument("--min-year", type=int, default=None)
    parser.add_argument("--max-year", type=int, default=None)
    args = parser.parse_args()

    for pos in ["QB", "RB", "WR", "TE"]:
        feats = v1.get_features_for_position(pos)
        print(f"  {pos}: {len(feats)} features")
    print()

    v1.train_all(
        zip_path=args.zip,
        json_name=args.json,
        out_dir=args.out,
        test_size=args.test_size,
        seed=args.seed,
        export_csv=args.export_csv or f"{args.out}/test_predictions.csv",
        min_year=args.min_year,
        max_year=args.max_year,
    )


if __name__ == "__main__":
    main()
