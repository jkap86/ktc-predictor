"""Populate team context features: qb_ktc, team_total_ktc, positional_competition.

Uses nfl_data_py seasonal rosters to get team assignments, then looks up
each teammate's KTC value from the ktc_dynasty postgres table at season start.

Usage:
    cd backend
    python -m scripts.populate_team_context
    python -m scripts.populate_team_context --dry-run
"""

import argparse
import asyncio
import json
import os
import zipfile
from collections import defaultdict
from datetime import date
from pathlib import Path

import asyncpg
import nfl_data_py as nfl
from dotenv import load_dotenv

load_dotenv()

POSITIONS = {"QB", "RB", "WR", "TE"}
# Approximate NFL season start dates for KTC lookups
SEASON_START_DATES = {
    2020: date(2020, 9, 1),
    2021: date(2021, 9, 1),
    2022: date(2022, 9, 1),
    2023: date(2023, 9, 1),
    2024: date(2024, 9, 1),
    2025: date(2025, 9, 1),
}


async def fetch_ktc_at_date(conn, player_ids: list[str], target_date: date) -> dict[str, float]:
    """Get KTC value closest to (but not after) target_date for each player."""
    if not player_ids:
        return {}
    rows = await conn.fetch(
        """
        SELECT DISTINCT ON (player_id) player_id, value
        FROM ktc_dynasty
        WHERE player_id = ANY($1) AND date <= $2
        ORDER BY player_id, date DESC
        """,
        player_ids,
        target_date,
    )
    return {r["player_id"]: float(r["value"]) for r in rows}


async def main_async(data_path: str, dry_run: bool):
    with open(data_path) as f:
        data = json.load(f)

    # Build sleeper_id -> player mapping
    sleeper_to_player = {p["player_id"]: p for p in data["players"]}

    print("Loading seasonal rosters from nfl_data_py...")
    years = list(range(2020, 2026))
    rosters = nfl.import_seasonal_rosters(years)
    rosters = rosters[rosters["position"].isin(POSITIONS)]
    rosters = rosters[rosters["sleeper_id"].notna()]
    rosters["sleeper_id"] = rosters["sleeper_id"].astype(int).astype(str)
    print(f"  {len(rosters)} roster entries across {len(years)} seasons")

    # Build team rosters per season: (year, team) -> list of {sleeper_id, position}
    team_rosters: dict[tuple[int, str], list[dict]] = defaultdict(list)
    for _, row in rosters.iterrows():
        team_rosters[(int(row["season"]), row["team"])].append({
            "sleeper_id": row["sleeper_id"],
            "position": row["position"],
        })

    # Map each player-season to their team
    # Use rosters to find: (sleeper_id, year) -> team
    player_team: dict[tuple[str, int], str] = {}
    for _, row in rosters.iterrows():
        player_team[(row["sleeper_id"], int(row["season"]))] = row["team"]

    print(f"  Player-team mappings: {len(player_team)}")

    # Connect to DB for KTC lookups
    db_url = os.getenv("DATABASE_URL")
    conn = await asyncpg.connect(db_url)

    filled = {"qb_ktc": 0, "team_total_ktc": 0, "positional_competition": 0}
    total_seasons = 0

    for year in years:
        target_date = SEASON_START_DATES.get(year, date(year, 9, 1))

        # Collect all sleeper_ids on rosters this year
        year_player_ids = set()
        for (y, team), members in team_rosters.items():
            if y != year:
                continue
            for m in members:
                year_player_ids.add(m["sleeper_id"])

        # Batch-fetch KTC values at season start
        ktc_values = await fetch_ktc_at_date(conn, list(year_player_ids), target_date)

        # Now compute team context for each player-season in training data
        for player in data["players"]:
            pid = player["player_id"]
            pos = player["position"]
            if pos not in POSITIONS:
                continue

            for season in player["seasons"]:
                if season["year"] != year:
                    continue
                total_seasons += 1

                team = player_team.get((pid, year))
                if not team:
                    continue

                teammates = team_rosters.get((year, team), [])
                if not teammates:
                    continue

                # QB KTC (highest KTC QB on the team)
                qb_ktcs = [
                    ktc_values.get(m["sleeper_id"], 0)
                    for m in teammates
                    if m["position"] == "QB" and m["sleeper_id"] != pid
                ]
                qb_ktc = max(qb_ktcs) if qb_ktcs else 0

                # Team total KTC (all skill positions)
                team_total = sum(
                    ktc_values.get(m["sleeper_id"], 0)
                    for m in teammates
                    if m["sleeper_id"] != pid
                )

                # Positional competition (same-position teammates' KTC)
                pos_competition = sum(
                    ktc_values.get(m["sleeper_id"], 0)
                    for m in teammates
                    if m["position"] == pos and m["sleeper_id"] != pid
                )

                if qb_ktc > 0:
                    season["qb_ktc"] = round(qb_ktc, 1)
                    filled["qb_ktc"] += 1
                if team_total > 0:
                    season["team_total_ktc"] = round(team_total, 1)
                    filled["team_total_ktc"] += 1
                if pos_competition > 0:
                    season["positional_competition"] = round(pos_competition, 1)
                    filled["positional_competition"] += 1

    await conn.close()

    print(f"\nResults ({total_seasons} player-seasons processed):")
    for k, v in filled.items():
        print(f"  {k}: {v} filled ({100*v/total_seasons:.1f}%)")

    if dry_run:
        print("\n[DRY RUN] No files written.")
    else:
        with open(data_path, "w") as f:
            json.dump(data, f)
        zip_path = Path(data_path).parent / "training-data.zip"
        with zipfile.ZipFile(zip_path, "w", zipfile.ZIP_DEFLATED) as zf:
            zf.writestr("training-data.json", json.dumps(data))
        print(f"\nWrote {data_path} and {zip_path}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", default="data/training-data.json")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    asyncio.run(main_async(args.data, args.dry_run))


if __name__ == "__main__":
    main()
