"""Train temporal model variants on expanding year windows.

Usage:
    cd backend
    python -m iterations.train_temporal --zip data/training-data.zip

Train a single variant:
    python -m iterations.train_temporal --zip data/training-data.zip --only v1_hgb_2021_2023
"""

import argparse
import os
import sys

from iterations.v1_hgb_baseline.train import train_all

TEMPORAL_VARIANTS = [
    {"id": "v1_hgb_2021",      "min_year": 2021, "max_year": 2021},
    {"id": "v1_hgb_2021_2022", "min_year": 2021, "max_year": 2022},
    {"id": "v1_hgb_2021_2023", "min_year": 2021, "max_year": 2023},
    {"id": "v1_hgb_2021_2024", "min_year": 2021, "max_year": 2024},
    {"id": "v1_hgb_2021_2025", "min_year": 2021, "max_year": 2025},
]

VARIANT_IDS = [v["id"] for v in TEMPORAL_VARIANTS]


def main():
    parser = argparse.ArgumentParser(
        description="Train all temporal model variants (expanding year windows)"
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
        "--model-type",
        default="hgb",
        choices=["hgb", "ridge", "elasticnet-poly"],
        help="Model type (default: hgb)",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed (default: 42)",
    )
    parser.add_argument(
        "--only",
        default=None,
        choices=VARIANT_IDS,
        help="Train only a single variant",
    )

    args = parser.parse_args()

    variants = TEMPORAL_VARIANTS
    if args.only:
        variants = [v for v in TEMPORAL_VARIANTS if v["id"] == args.only]

    print(f"Training {len(variants)} temporal variant(s)...")
    print()

    for i, variant in enumerate(variants, 1):
        vid = variant["id"]
        out_dir = f"iterations/{vid}/models"

        print("=" * 70)
        print(f"  [{i}/{len(variants)}] {vid}  (years {variant['min_year']}-{variant['max_year']})")
        print("=" * 70)
        print()

        os.makedirs(out_dir, exist_ok=True)

        try:
            train_all(
                zip_path=args.zip,
                json_name=args.json,
                out_dir=out_dir,
                model_type=args.model_type,
                seed=args.seed,
                min_year=variant["min_year"],
                max_year=variant["max_year"],
                export_csv=f"{out_dir}/test_predictions.csv",
            )
            print(f"  Completed {vid}")
        except Exception as e:
            print(f"  FAILED {vid}: {e}", file=sys.stderr)

        print()

    print("=" * 70)
    print("  All temporal variants complete.")
    print("=" * 70)


if __name__ == "__main__":
    main()
