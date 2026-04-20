"""Populate draft_pick and years_remaining in training-data.json.

Uses nfl_data_py for:
- Draft picks: maps via Sleeper ID -> GSIS ID -> draft pick number
- Contract years_remaining: maps via Sleeper ID -> GSIS ID -> contract data

Usage:
    cd backend
    python -m scripts.populate_draft_contract
    python -m scripts.populate_draft_contract --dry-run  # preview without writing
"""

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
import nfl_data_py as nfl


def load_id_mapping() -> dict[str, str]:
    """Sleeper ID (str) -> GSIS ID (str) mapping."""
    ids = nfl.import_ids()
    valid = ids[ids["sleeper_id"].notna() & ids["gsis_id"].notna()].copy()
    valid["sleeper_id_str"] = valid["sleeper_id"].astype(int).astype(str)
    return dict(zip(valid["sleeper_id_str"], valid["gsis_id"]))


def load_draft_picks() -> dict[str, int]:
    """GSIS ID -> overall draft pick number."""
    drafts = nfl.import_draft_picks(list(range(2000, 2026)))
    drafts = drafts[drafts["gsis_id"].notna() & drafts["pick"].notna()]
    # Deduplicate: keep earliest (first draft entry)
    drafts = drafts.sort_values("season").drop_duplicates("gsis_id", keep="first")
    return dict(zip(drafts["gsis_id"], drafts["pick"].astype(int)))


def load_contracts() -> pd.DataFrame:
    """Contract data with years_remaining computable per season."""
    contracts = nfl.import_contracts()
    contracts = contracts[contracts["position"].isin(["QB", "RB", "WR", "TE"])]
    contracts = contracts[contracts["gsis_id"].notna()]
    return contracts[["gsis_id", "year_signed", "years", "value", "apy"]].copy()


def compute_years_remaining(contracts: pd.DataFrame, gsis_id: str, season_year: int) -> float | None:
    """Compute years remaining on contract at the start of a given season."""
    player_contracts = contracts[contracts["gsis_id"] == gsis_id]
    if player_contracts.empty:
        return None

    # Find the active contract for this season
    for _, row in player_contracts.sort_values("year_signed", ascending=False).iterrows():
        signed = int(row["year_signed"]) if pd.notna(row["year_signed"]) else 0
        length = int(row["years"]) if pd.notna(row["years"]) else 0
        if signed <= season_year < signed + length:
            return float(signed + length - season_year)

    return None


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", default="data/training-data.json")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    data_path = Path(args.data)
    with open(data_path) as f:
        data = json.load(f)

    print("Loading nfl_data_py mappings...")
    sleeper_to_gsis = load_id_mapping()
    print(f"  ID mapping: {len(sleeper_to_gsis)} Sleeper -> GSIS entries")

    gsis_to_draft = load_draft_picks()
    print(f"  Draft picks: {len(gsis_to_draft)} players")

    contracts = load_contracts()
    print(f"  Contracts: {len(contracts)} entries")

    draft_filled = 0
    years_filled = 0
    total_seasons = 0

    for player in data["players"]:
        pid = player["player_id"]
        gsis_id = sleeper_to_gsis.get(pid)

        # Draft pick (same for all seasons)
        draft_pick = None
        if gsis_id:
            draft_pick = gsis_to_draft.get(gsis_id)

        for season in player.get("seasons", []):
            total_seasons += 1

            # Draft pick
            if draft_pick is not None and not season.get("draft_pick"):
                season["draft_pick"] = draft_pick
                draft_filled += 1

            # Years remaining
            if gsis_id and not season.get("years_remaining"):
                yr = compute_years_remaining(contracts, gsis_id, season["year"])
                if yr is not None:
                    season["years_remaining"] = yr
                    years_filled += 1

    print(f"\nResults:")
    print(f"  Total player-seasons: {total_seasons}")
    print(f"  draft_pick filled: {draft_filled}")
    print(f"  years_remaining filled: {years_filled}")

    # Check coverage
    has_draft = sum(1 for p in data["players"] for s in p["seasons"] if s.get("draft_pick"))
    has_years = sum(1 for p in data["players"] for s in p["seasons"] if s.get("years_remaining"))
    print(f"  draft_pick coverage: {has_draft}/{total_seasons} ({100*has_draft/total_seasons:.1f}%)")
    print(f"  years_remaining coverage: {has_years}/{total_seasons} ({100*has_years/total_seasons:.1f}%)")

    if args.dry_run:
        print("\n[DRY RUN] No files written.")
    else:
        # Write updated JSON
        with open(data_path, "w") as f:
            json.dump(data, f)
        print(f"\nWrote {data_path}")

        # Also update the zip
        import zipfile
        zip_path = data_path.parent / "training-data.zip"
        with zipfile.ZipFile(zip_path, "w", zipfile.ZIP_DEFLATED) as zf:
            zf.writestr("training-data.json", json.dumps(data))
        print(f"Wrote {zip_path}")


if __name__ == "__main__":
    main()
