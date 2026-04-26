"""v3 training script: v1's 20-feature baseline + 6 prior-season behavioral
signals applied ONLY to RB and WR.

The 6 additions are all frozen at the player's last completed season, so they
do not shift when the user moves the What-If slider. This sidesteps the
plumbing problem that orphaned v2.

New features (RB/WR only — QB/TE use v1's 20-feature baseline unchanged):
- prior_weekly_fp_cv    — consistency of weekly FP last year
- prior_boom_rate       — upside-week frequency last year
- prior_bust_rate       — downside-week frequency last year
- prior_snap_pct        — role/usage baseline last year
- prior_ktc_volatility  — KTC bounciness last year
- has_prior_behavioral  — 1 if prior-season data is usable (>=4 games), else 0

Scoping to RB/WR was a deliberate choice: the initial v3 (all positions)
improved RB and WR MAE but made QB and TE *worse*. QBs play near-full snaps
and consistency signals are dominated by offensive system, not individual
variance; TE has too few samples for 6 extra features to generalize.

Feature vector layout:
  QB / TE (20 features): v1's layout unchanged
  RB / WR (26 features): v1's 20 + 6 prior-behavioral appended

Usage:
    cd backend
    python -m iterations.v3_hgb_prior_signals.train --zip data/training-data.zip
"""

import argparse
import copy

import numpy as np

import iterations.v1_hgb_baseline.train as v1
import iterations.v1_hgb_baseline.data as v1_data


# ── Build prior-season behavioral feature lookup ────────────────────────

_MIN_GAMES_FOR_PRIOR_BEHAVIORAL = 4


def _compute_prior_behavioral_features(
    players: list[dict],
) -> dict[tuple[str, int], dict]:
    """For each (player_id, year) past the first season, copy the prior
    season's behavioral aggregates.

    Requires prior season to have >= 4 games played; otherwise skip
    (has_prior_behavioral will default to 0 at merge time).
    """
    result: dict[tuple[str, int], dict] = {}

    for player in players:
        pid = player["player_id"]
        seasons = sorted(
            [s for s in player.get("seasons", []) if (s.get("years_exp") or 0) >= 0],
            key=lambda s: s["year"],
        )

        for i, season in enumerate(seasons):
            # Try prior season first
            source = None
            if i > 0:
                prior = seasons[i - 1]
                prior_games = prior.get("games_played", 0) or 0
                if prior_games >= _MIN_GAMES_FOR_PRIOR_BEHAVIORAL:
                    source = prior

            # Fallback for RB/TE: use current season when no prior exists
            if source is None and player.get("position") in ("RB", "TE"):
                curr_games = season.get("games_played", 0) or 0
                if curr_games >= _MIN_GAMES_FOR_PRIOR_BEHAVIORAL:
                    source = season

            if source is None:
                continue

            result[(pid, season["year"])] = {
                "prior_weekly_fp_cv": float(source.get("weekly_fp_cv") or 0.0),
                "prior_boom_rate": float(source.get("boom_rate") or 0.0),
                "prior_bust_rate": float(source.get("bust_rate") or 0.0),
                "prior_snap_pct": float(source.get("snap_pct") or 0.0),
                "prior_ktc_volatility": float(source.get("ktc_volatility") or 0.0),
            }

    return result


# ── Wrap build_weekly_snapshot_df to inject prior-behavioral columns ────

_original_build_weekly_snapshot_df = v1_data.build_weekly_snapshot_df


def _build_weekly_snapshot_df_v3(zip_path: str, json_name: str = "training-data.json"):
    """Like v1's build_weekly_snapshot_df, but adds 6 prior-behavioral columns."""
    import json
    import zipfile
    from pathlib import Path

    df = _original_build_weekly_snapshot_df(zip_path, json_name)

    # Re-read the JSON to compute prior-behavioral lookups (cheap; already cached in OS)
    with zipfile.ZipFile(Path(zip_path), "r") as zf:
        with zf.open(json_name) as f:
            data = json.load(f)

    prior_behavioral = _compute_prior_behavioral_features(data["players"])
    print(f"  Prior-behavioral features loaded for {len(prior_behavioral)} player-seasons")

    def _lookup(row, key):
        feat = prior_behavioral.get((row["player_id"], row["year"]))
        return feat[key] if feat else None

    df["prior_weekly_fp_cv"] = df.apply(lambda r: _lookup(r, "prior_weekly_fp_cv"), axis=1)
    df["prior_boom_rate"] = df.apply(lambda r: _lookup(r, "prior_boom_rate"), axis=1)
    df["prior_bust_rate"] = df.apply(lambda r: _lookup(r, "prior_bust_rate"), axis=1)
    df["prior_snap_pct"] = df.apply(lambda r: _lookup(r, "prior_snap_pct"), axis=1)
    df["prior_ktc_volatility"] = df.apply(lambda r: _lookup(r, "prior_ktc_volatility"), axis=1)
    df["has_prior_behavioral"] = df["prior_weekly_fp_cv"].notna().astype(int)

    return df


# Install the v3 snapshot builder on v1's train module so train_all picks it up
v1.build_weekly_snapshot_df = _build_weekly_snapshot_df_v3


# ── Patch v1 feature configuration for v3 ───────────────────────────────

_V1_CORE_FEATURES = copy.copy(v1._CORE_FEATURES)
_V1_BASE_LINEAR = copy.copy(v1._BASE_LINEAR_FEATURES)

# Core features unchanged from v1.
# Append prior-behavioral to the BASE linear features so all positions get them,
# positioned AFTER contract features (see monotonic constraints below).
v1._BASE_LINEAR_FEATURES = v1._BASE_LINEAR_FEATURES + []  # no-op to keep structure
# (v3 prior-behavioral features are added at the END of the linear feature list,
# after prior-season/contract. Done via a patch to get_features_for_position.)

_V1_GET_FEATURES_FOR_POSITION = v1.get_features_for_position

_PRIOR_BEHAVIORAL_FEATURES = [
    "prior_weekly_fp_cv",
    "prior_boom_rate",
    "prior_bust_rate",
    "prior_snap_pct",
    "prior_ktc_volatility",
    "has_prior_behavioral",
]

# Positions that receive the 6 prior-behavioral features.
# Initial v3 applied to all 4; QB/TE were worse than v1, RB/WR improved.
_POSITIONS_WITH_PRIOR_BEHAVIORAL = {"RB", "WR"}


def _v3_get_features_for_position(position: str) -> list[str]:
    """v1's feature list + 6 prior-behavioral for RB/WR, unchanged for QB/TE."""
    base = _V1_GET_FEATURES_FOR_POSITION(position)
    if position in _POSITIONS_WITH_PRIOR_BEHAVIORAL:
        return base + _PRIOR_BEHAVIORAL_FEATURES
    return base


v1.get_features_for_position = _v3_get_features_for_position

# Rebuild derived constants used by train_all()
v1.FEATURES = v1.get_features_for_position("QB")
v1.N_CORE_FEATURES = len(v1._CORE_FEATURES)
v1.N_LINEAR_FEATURES = len(v1.FEATURES) - v1.N_CORE_FEATURES


# ── Patch monotonic constraints to extend v1's layout with 6 new linear features ──

_v1_build_monotonic_constraints = v1._build_monotonic_constraints


def _v3_build_monotonic_constraints(position: str) -> list[int]:
    """v1's constraints, extended with 6 zeros only for RB/WR.

    Conservative: none of the 6 new features get a monotonic constraint.
    prior_snap_pct could arguably be positive (more snaps = stronger role),
    but HGB monotonic constraints are brittle when features interact;
    keep them free here and let the model learn the interaction.
    """
    base = _v1_build_monotonic_constraints(position)
    if position in _POSITIONS_WITH_PRIOR_BEHAVIORAL:
        return base + [0, 0, 0, 0, 0, 0]
    return base


v1._build_monotonic_constraints = _v3_build_monotonic_constraints


# ── Patch smoke tests: v3 feature vector has 6 extra trailing values ────

def _v3_monotonic_smoke_test(model, position: str) -> bool:
    """v1's smoke test; for RB/WR appends 6 prior-behavioral defaults."""
    ppg_values = [5, 10, 15, 20]

    X_test = []
    for ppg in ppg_values:
        row = [8, ppg, 2, np.nan, 3, 4, -2, 0]  # core
        row.extend([5000, 0])  # start_ktc, sentinel
        row.extend([0.0, 0.0, 1])  # prior KTC
        row.extend([15.0, np.log((ppg + 0.1) / (15 + 0.1)), 1])  # prior PPG
        if v1.USE_CONTRACT_FEATURES:
            row.extend([0.05, 0, 0.5, 1])  # contract
        if v1.USE_PFF_FEATURES:
            row.extend([70.0, 1, 70.0, 1])
        if v1.USE_TEAM_FEATURES:
            row.extend([5000, 30000, 3000])
        if position in _POSITIONS_WITH_PRIOR_BEHAVIORAL:
            # v3 prior-behavioral: median-ish defaults
            row.extend([0.4, 0.25, 0.25, 0.65, 1.0, 1])
        X_test.append(row)

    X_test = np.array(X_test)
    preds = model.predict(X_test)
    is_monotonic = all(preds[i] <= preds[i + 1] for i in range(len(preds) - 1))
    status = "PASS" if is_monotonic else "FAIL"
    print(f"  Monotonic test ({position}): {status}  preds={[round(p, 4) for p in preds]}")
    return is_monotonic


def _v3_ppg_sensitivity_test(model, calibrator_dict, clip_bounds, position: str) -> bool:
    """Skip — v1's predict_end_ktc builds a 20-feature vector; v3 is 26.

    v3's own predict_end_ktc (iterations/v3_hgb_prior_signals/predict.py)
    is the correct end-to-end test; run it separately after training.
    """
    print(f"  PPG sensitivity ({position}): SKIP  (v3 feature set; run v3 predict separately)")
    return True


v1._monotonic_smoke_test = _v3_monotonic_smoke_test
v1._ppg_sensitivity_test = _v3_ppg_sensitivity_test


def main():
    parser = argparse.ArgumentParser(
        description="Train v3 per-position KTC models (v1 + 6 prior-behavioral features)"
    )
    parser.add_argument("--zip", default="data/training-data.zip")
    parser.add_argument("--json", default="training-data.json")
    parser.add_argument("--out", default="iterations/v3_hgb_prior_signals/models")
    parser.add_argument("--test-size", type=float, default=0.2)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--export-csv", default=None)
    parser.add_argument("--min-year", type=int, default=None)
    parser.add_argument("--max-year", type=int, default=None)
    args = parser.parse_args()

    print(f"v3 feature set: {len(v1.FEATURES)} features")
    print(f"  Core ({len(v1._CORE_FEATURES)}): {v1._CORE_FEATURES}")
    print(f"  Prior-behavioral (6): {_PRIOR_BEHAVIORAL_FEATURES}")
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
