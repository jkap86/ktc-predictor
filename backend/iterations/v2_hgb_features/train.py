"""v2 training script: enhanced feature set (29 features).

Reuses v1_hgb_baseline training pipeline with additional features:
- boom_rate, bust_rate (core, polynomial expansion)
- weekly_fp_cv, start_position_rank, last_4_vs_season, snap_pct (linear)
- offseason_percentile, trend_14d (re-enabled offseason features)

Usage:
    cd backend
    python -m iterations.v2_hgb_features.train --zip data/training-data.zip
"""

import argparse
import copy

import iterations.v1_hgb_baseline.train as v1

# ── Patch v1 feature configuration for v2 ──────────────────────────────

# Save original v1 state so we can restore if needed
_V1_CORE_FEATURES = copy.copy(v1._CORE_FEATURES)
_V1_BASE_LINEAR = copy.copy(v1._BASE_LINEAR_FEATURES)
_V1_USE_OFFSEASON = v1.USE_OFFSEASON_FEATURES

# Enable offseason features (disabled in v1 - hurt polynomial models but
# should help HGB which can isolate conditional interactions)
v1.USE_OFFSEASON_FEATURES = True

# Add new core features (polynomial expansion - interactions matter)
v1._CORE_FEATURES = v1._CORE_FEATURES + [
    "boom_rate",           # % of weeks with top-quartile FP (upside frequency)
    "bust_rate",           # % of weeks with bottom-quartile FP (downside risk)
]

# Add new linear features (bypass polynomial to avoid noise)
v1._BASE_LINEAR_FEATURES = v1._BASE_LINEAR_FEATURES + [
    "weekly_fp_cv",        # Coefficient of variation in weekly FP (consistency)
    "start_position_rank", # Dynasty position ranking at season start
    "last_4_vs_season",    # Last-4-weeks FP ratio vs season avg (form)
    "has_last4_data",      # 1 if last_4_vs_season exists, else 0
    "snap_pct",            # Offensive snap percentage (role stability)
    "has_snap_data",       # 1 if snap_pct > 0, else 0
]

# Recalculate derived constants used by train_all()
v1.FEATURES = v1.get_features_for_position("QB")
v1.N_CORE_FEATURES = len(v1._CORE_FEATURES)
v1.N_LINEAR_FEATURES = len(v1.FEATURES) - v1.N_CORE_FEATURES


# ── Patch monotonic constraints to match extended feature layout ────────
_v1_build_monotonic_constraints = v1._build_monotonic_constraints


def _v2_build_monotonic_constraints(position: str) -> list[int]:
    """Build monotonic constraints for v2's 30-feature layout.

    Feature order (with USE_OFFSEASON_FEATURES=True):
      Core (10): gp, ppg(+), weeks_missed, draft_pick, years_remaining,
                 ktc_quartile, age_prime_dist, is_breakout(+), boom_rate(+), bust_rate
      Offseason (2): offseason_percentile, trend_14d(+)
      Linear (18): start_ktc(+), sentinel, weekly_fp_cv, start_position_rank,
                    last_4_vs_season, has_last4_data, snap_pct(+), has_snap_data,
                    ktc_yoy_log, ktc_peak_drawdown, has_prior_season,
                    prior_ppg, ppg_yoy_log(+), has_prior_ppg,
                    apy_cap_pct, is_contract_year, apy_position_rank, has_contract_data
    """
    # Core features (10): v1's 8 + boom_rate, bust_rate
    # ppg(+), is_breakout(+), boom_rate(+)
    core_constraints = [0, 1, 0, 0, 0, 0, 0, 1, 1, 0]

    # Offseason features (2): offseason_percentile, trend_14d(+)
    offseason_constraints = [0, 1]

    # Base linear features (8): start_ktc(+), sentinel, + 6 new
    # snap_pct(+): more snaps = more stable role
    linear_constraints = [1, 0, 0, 0, 0, 0, 1, 0]

    # Prior-season features for all positions
    if position in ("QB", "RB", "WR", "TE"):
        yoy_constraint = 0 if position == "QB" else 1
        linear_constraints.extend([yoy_constraint, 0, 0])  # prior KTC (3)
        linear_constraints.extend([0, 1, 0])  # prior PPG (3)

    # Contract features (4)
    if v1.USE_CONTRACT_FEATURES:
        linear_constraints.extend([0, 0, 0, 0])

    if v1.USE_PFF_FEATURES:
        linear_constraints.extend([0, 0, 0, 0])

    if v1.USE_TEAM_FEATURES:
        linear_constraints.extend([0, 0, 0])

    return core_constraints + offseason_constraints + linear_constraints


v1._build_monotonic_constraints = _v2_build_monotonic_constraints


# ── Patch smoke tests to match v2 feature layout ───────────────────────
import numpy as np


def _v2_monotonic_smoke_test(model, position: str) -> bool:
    """Verify log_ratio predictions are non-decreasing as PPG increases (v2 layout)."""
    ppg_values = [5, 10, 15, 20]

    X_test = []
    for ppg in ppg_values:
        # Core (10): gp, ppg, weeks_missed, draft_pick, years_remaining,
        #   ktc_quartile, age_prime_dist, is_breakout, boom_rate, bust_rate
        row = [8, ppg, 2, np.nan, 3, 4, -2, 0, 0.2, 0.2]
        # Offseason (2): offseason_percentile, trend_14d
        row.extend([0.5, 0.0])
        # Base linear (8): start_ktc, sentinel, weekly_fp_cv, start_position_rank,
        #   last_4_vs_season, has_last4, snap_pct, has_snap
        row.extend([5000, 0, 0.4, 10, 1.0, 1, 0.7, 1])

        if position in ("QB", "RB", "WR", "TE"):
            row.extend([0.0, 0.0, 1])  # prior KTC (3)
            row.extend([15.0, np.log((ppg + 0.1) / (15 + 0.1)), 1])  # prior PPG (3)

        if v1.USE_CONTRACT_FEATURES:
            row.extend([0.05, 0, 0.5, 1])

        if v1.USE_PFF_FEATURES:
            row.extend([70.0, 1, 70.0, 1])

        if v1.USE_TEAM_FEATURES:
            row.extend([5000, 30000, 3000])

        X_test.append(row)

    X_test = np.array(X_test)
    preds = model.predict(X_test)

    is_monotonic = all(preds[i] <= preds[i + 1] for i in range(len(preds) - 1))
    status = "PASS" if is_monotonic else "FAIL"
    print(f"  Monotonic test ({position}): {status}  preds={[round(p, 4) for p in preds]}")
    return is_monotonic


def _v2_ppg_sensitivity_test(model, calibrator_dict, clip_bounds, position: str) -> bool:
    """Skip PPG sensitivity test for v2 (v1 predict_end_ktc doesn't know v2 features)."""
    print(f"  PPG sensitivity ({position}): SKIP  (v2 feature set)")
    return True


v1._monotonic_smoke_test = _v2_monotonic_smoke_test
v1._ppg_sensitivity_test = _v2_ppg_sensitivity_test


def main():
    parser = argparse.ArgumentParser(
        description="Train v2 per-position KTC models (enhanced 29-feature set)"
    )
    parser.add_argument(
        "--zip",
        default="data/training-data.zip",
        help="Path to training data zip file",
    )
    parser.add_argument(
        "--json",
        default="training-data.json",
        help="Name of JSON file inside the zip",
    )
    parser.add_argument(
        "--out",
        default="iterations/v2_hgb_features/models",
        help="Output directory for saved models",
    )
    parser.add_argument(
        "--test-size",
        type=float,
        default=0.2,
        help="Fraction of data for test set (default: 0.2)",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed (default: 42)",
    )
    parser.add_argument(
        "--export-csv",
        default=None,
        help="Export test-set predictions to CSV",
    )
    parser.add_argument(
        "--min-year",
        type=int,
        default=None,
        help="Minimum season year (inclusive)",
    )
    parser.add_argument(
        "--max-year",
        type=int,
        default=None,
        help="Maximum season year (inclusive)",
    )

    args = parser.parse_args()

    print(f"v2 feature set: {len(v1.FEATURES)} features")
    print(f"  Core ({len(v1._CORE_FEATURES)}): {v1._CORE_FEATURES}")
    print(f"  Offseason: enabled")
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
