"""Add rookie/new players from the KTC database to training-data.json.

Finds players in ktc_dynasty that aren't in the training data and adds
them with a single 2026 season (gp=0, start_ktc from DB). This makes
them searchable in the UI and gives them predictions based on their
KTC value, draft pick, and age.

Usage:
    cd backend
    python -m scripts.add_rookies
    python -m scripts.add_rookies --dry-run
"""

import argparse
import asyncio
import json
import zipfile
from pathlib import Path

import urllib.request


DATA_PATH = Path("data/training-data.zip")
JSON_NAME = "training-data.json"


def load_sleeper_players() -> dict[str, dict]:
    """Fetch all NFL players from Sleeper API. Returns {id: {name, position, age, ...}}."""
    url = "https://api.sleeper.app/v1/players/nfl"
    req = urllib.request.Request(url)
    with urllib.request.urlopen(req, timeout=30) as resp:
        raw = json.loads(resp.read())

    result = {}
    for pid, p in raw.items():
        pos = p.get("position")
        name = p.get("full_name") or p.get("last_name") or "Unknown"
        if not pos:
            continue
        draft_meta = p.get("draft_metadata") or {}
        result[pid] = {
            "name": name,
            "position": pos,
            "age": p.get("age"),
            "draft_number": draft_meta.get("pick_no"),
            "team": p.get("team"),
        }
    return result


async def get_new_players_from_db(existing_ids: set[str]) -> list[dict]:
    """Query ktc_dynasty for players not in existing_ids."""
    import asyncpg
    import os
    from dotenv import load_dotenv
    load_dotenv()

    db_url = os.getenv("DATABASE_URL")
    if not db_url:
        raise RuntimeError("DATABASE_URL not set in .env")

    pool = await asyncpg.create_pool(db_url, min_size=1, max_size=2)
    try:
        rows = await pool.fetch("""
            SELECT DISTINCT ON (player_id) player_id, value, position_rank, date
            FROM ktc_dynasty
            ORDER BY player_id, date DESC
        """)
        new_players = []
        for r in rows:
            pid = str(r["player_id"])
            if pid not in existing_ids and r["value"] > 0:
                new_players.append({
                    "player_id": pid,
                    "ktc": int(r["value"]),
                    "position_rank": int(r["position_rank"]) if r["position_rank"] else None,
                    "date": str(r["date"]),
                })
        return new_players
    finally:
        await pool.close()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--year", type=int, default=2026, help="Season year for rookies")
    args = parser.parse_args()

    print("Loading training data...")
    with zipfile.ZipFile(DATA_PATH) as zf:
        with zf.open(JSON_NAME) as f:
            data = json.load(f)

    existing_ids = set(p["player_id"] for p in data["players"])
    print(f"  Existing players: {len(existing_ids)}")

    print("Loading player info from Sleeper API...")
    sleeper_info = load_sleeper_players()
    print(f"  Sleeper players: {len(sleeper_info)}")

    print("Querying database for new players...")
    new_players = asyncio.run(get_new_players_from_db(existing_ids))
    print(f"  New players found: {len(new_players)}")

    added = 0
    skipped_no_info = 0
    skipped_non_skill = 0

    for np_data in sorted(new_players, key=lambda x: -x["ktc"]):
        pid = np_data["player_id"]
        info = sleeper_info.get(pid)

        if not info:
            skipped_no_info += 1
            continue

        pos = info.get("position", "")
        if pos not in ("QB", "RB", "WR", "TE"):
            skipped_non_skill += 1
            continue

        age = info.get("age")
        draft_pick = info.get("draft_number")

        player = {
            "player_id": pid,
            "name": info["name"],
            "position": pos,
            "seasons": [
                {
                    "year": args.year,
                    "age": int(age) if age and str(age) != "nan" else None,
                    "years_exp": 0,
                    "games_played": 0,
                    "fantasy_points": 0,
                    "start_ktc": np_data["ktc"],
                    "end_ktc": np_data["ktc"],  # placeholder
                    "start_position_rank": np_data["position_rank"],
                    "draft_pick": int(draft_pick) if draft_pick and str(draft_pick) != "nan" else None,
                    "ktc_volatility": 0,
                    "weekly_stats": [],
                    "weekly_ktc": [],
                }
            ],
        }

        data["players"].append(player)
        added += 1

        if added <= 10:
            print(f"  + {info['name']:25s} {pos:3s} KTC={np_data['ktc']:>5} age={age} draft={draft_pick}")

    if added > 10:
        print(f"  ... and {added - 10} more")

    print(f"\nResults:")
    print(f"  Added: {added}")
    print(f"  Skipped (no Sleeper info): {skipped_no_info}")
    print(f"  Skipped (non-skill position): {skipped_non_skill}")

    if args.dry_run:
        print("\n  [DRY RUN — not saving]")
        return

    import tempfile
    print("\nSaving...")
    with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as tmp:
        json.dump(data, tmp, indent=None, separators=(",", ":"))
        tmp_path = tmp.name

    with zipfile.ZipFile(DATA_PATH, "w", zipfile.ZIP_DEFLATED) as zf:
        zf.write(tmp_path, JSON_NAME)
    Path(tmp_path).unlink()
    print(f"  Saved to {DATA_PATH} ({len(data['players'])} total players)")


if __name__ == "__main__":
    main()
