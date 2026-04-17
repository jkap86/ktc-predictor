"""Train temporal v3 model variants: train on ≤N-1, predict season N.

Each variant is a v3 model (with RB/WR prior-behavioral features) trained
on all data up through year N-1, designed to predict season N outcomes.
This lets users "go back in time" to see what the model would have predicted.

The models are stored as regular iterations in the registry so the existing
predict pipeline works unchanged. Their manifest includes a `predict_year`
field to indicate which season they target.

Usage:
    cd backend
    python -m iterations.train_temporal_v3 --zip data/training-data.zip

Train a single year:
    python -m iterations.train_temporal_v3 --zip data/training-data.zip --only 2023
"""

import argparse
import json
import os
import shutil
import sys

# Import v3's train module — this applies all the monkey-patches to v1
import iterations.v3_hgb_prior_signals.train  # noqa: F401 (side-effect: patches v1)
import iterations.v1_hgb_baseline.train as v1

TEMPORAL_VARIANTS = [
    {"id": "v3_for_2021", "max_year": 2020, "predict_year": 2021},
    {"id": "v3_for_2022", "max_year": 2021, "predict_year": 2022},
    {"id": "v3_for_2023", "max_year": 2022, "predict_year": 2023},
    {"id": "v3_for_2024", "max_year": 2023, "predict_year": 2024},
    {"id": "v3_for_2025", "max_year": 2024, "predict_year": 2025},
]


def main():
    parser = argparse.ArgumentParser(
        description="Train temporal v3 model variants (train on ≤N-1, predict season N)"
    )
    parser.add_argument("--zip", default="data/training-data.zip")
    parser.add_argument("--json", default="training-data.json")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--only", type=int, default=None,
        help="Train only for a single predict year (e.g. 2023)",
    )
    args = parser.parse_args()

    variants = TEMPORAL_VARIANTS
    if args.only:
        variants = [v for v in TEMPORAL_VARIANTS if v["predict_year"] == args.only]
        if not variants:
            print(f"No variant for predict_year={args.only}")
            sys.exit(1)

    print(f"Training {len(variants)} temporal v3 variant(s)...")
    print()

    for i, variant in enumerate(variants, 1):
        vid = variant["id"]
        iteration_dir = f"iterations/{vid}"
        out_dir = f"{iteration_dir}/models"

        print("=" * 70)
        print(f"  [{i}/{len(variants)}] {vid}  (train <={variant['max_year']}, predict {variant['predict_year']})")
        print("=" * 70)
        print()

        os.makedirs(out_dir, exist_ok=True)

        # Write manifest
        manifest = {
            "id": vid,
            "name": f"v3 for {variant['predict_year']}",
            "description": f"v3 model trained on ≤{variant['max_year']} to predict {variant['predict_year']} season outcomes",
            "created": "2026-04-17",
            "predict_year": variant["predict_year"],
            "train_max_year": variant["max_year"],
            "features_count": {"QB": 20, "RB": 26, "WR": 26, "TE": 20},
            "positions": ["QB", "RB", "WR", "TE"],
            "temporal": True,
        }
        with open(f"{iteration_dir}/manifest.json", "w") as f:
            json.dump(manifest, f, indent=2)

        # Copy v3's predict.py and io.py shims
        v3_dir = "iterations/v3_hgb_prior_signals"
        for fname in ("__init__.py", "predict.py", "io.py"):
            src = os.path.join(v3_dir, fname)
            dst = os.path.join(iteration_dir, fname)
            if os.path.exists(src) and not os.path.exists(dst):
                shutil.copy2(src, dst)

        try:
            v1.train_all(
                zip_path=args.zip,
                json_name=args.json,
                out_dir=out_dir,
                seed=args.seed,
                max_year=variant["max_year"],
                export_csv=f"{out_dir}/test_predictions.csv",
            )
            print(f"  Completed {vid}")
        except Exception as e:
            print(f"  FAILED {vid}: {e}", file=sys.stderr)

        print()

    print("=" * 70)
    print("  All temporal v3 variants complete.")
    print("=" * 70)


if __name__ == "__main__":
    main()
