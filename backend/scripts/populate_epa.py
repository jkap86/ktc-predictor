"""Populate EPA and efficiency metrics in training-data.json from nfl_data_py.

Adds per-season fields:
  - rushing_epa: total rushing EPA for the season
  - receiving_epa: total receiving EPA for the season
  - target_share: share of team targets
  - dominator_rating: (from nfl_data_py 'dom' field)

These are used as prior-season features to differentiate players
at similar KTC/PPG levels (especially young RBs with sparse prior data).

Usage:
    cd backend
    python -m scripts.populate_epa
    python -m scripts.populate_epa --dry-run
"""

import argparse
import json
from pathlib import Path

import nfl_data_py as nfl


DATA_PATH = Path("data/training-data.zip")
JSON_NAME = "training-data.json"


def load_id_mapping() -> dict[str, str]:
    """Sleeper ID (str) -> GSIS ID (str) mapping."""
    ids = nfl.import_ids()
    valid = ids[ids["sleeper_id"].notna() & ids["gsis_id"].notna()].copy()
    valid["sleeper_id_str"] = valid["sleeper_id"].astype(int).astype(str)
    return dict(zip(valid["sleeper_id_str"], valid["gsis_id"]))


def load_seasonal_epa(years: list[int]) -> dict[tuple[str, int], dict]:
    """GSIS ID + season -> EPA metrics dict."""
    df = nfl.import_seasonal_data(years, "REG")

    result = {}
    for _, row in df.iterrows():
        gsis = row["player_id"]
        year = int(row["season"])
        if not gsis:
            continue
        result[(gsis, year)] = {
            "rushing_epa": _safe_float(row.get("rushing_epa")),
            "receiving_epa": _safe_float(row.get("receiving_epa")),
            "target_share": _safe_float(row.get("target_share")),
            "dominator_rating": _safe_float(row.get("dom")),
        }
    return result


def _safe_float(val) -> float | None:
    if val is None:
        return None
    try:
        f = float(val)
        if f != f:  # NaN check
            return None
        return round(f, 4)
    except (ValueError, TypeError):
        return None


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    import zipfile

    print("Loading ID mapping...")
    sleeper_to_gsis = load_id_mapping()
    print(f"  {len(sleeper_to_gsis)} Sleeper->GSIS mappings")

    print("Loading seasonal EPA data...")
    epa_data = load_seasonal_epa(list(range(2020, 2025)))
    print(f"  {len(epa_data)} player-seasons with EPA data")

    print("Loading training data...")
    with zipfile.ZipFile(DATA_PATH) as zf:
        with zf.open(JSON_NAME) as f:
            data = json.load(f)

    players = data["players"]
    total_seasons = 0
    populated = 0
    no_gsis = 0
    no_epa = 0

    for player in players:
        gsis = sleeper_to_gsis.get(player["player_id"])
        if not gsis:
            no_gsis += len(player.get("seasons", []))
            continue

        for season in player.get("seasons", []):
            total_seasons += 1
            year = season["year"]
            epa = epa_data.get((gsis, year))
            if epa:
                season["rushing_epa"] = epa["rushing_epa"]
                season["receiving_epa"] = epa["receiving_epa"]
                season["target_share"] = epa["target_share"]
                season["dominator_rating"] = epa["dominator_rating"]
                populated += 1
            else:
                no_epa += 1

    print(f"\nResults:")
    print(f"  Total seasons: {total_seasons}")
    print(f"  Populated: {populated} ({populated/max(total_seasons,1)*100:.1f}%)")
    print(f"  No GSIS ID: {no_gsis}")
    print(f"  No EPA data: {no_epa}")

    if args.dry_run:
        print("\n  [DRY RUN — not saving]")
        return

    # Write back
    import tempfile, shutil

    print("\nSaving...")
    with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as tmp:
        json.dump(data, tmp, indent=None, separators=(",", ":"))
        tmp_path = tmp.name

    # Replace zip
    with zipfile.ZipFile(DATA_PATH, "w", zipfile.ZIP_DEFLATED) as zf:
        zf.write(tmp_path, JSON_NAME)
    Path(tmp_path).unlink()
    print(f"  Saved to {DATA_PATH}")


if __name__ == "__main__":
    main()
