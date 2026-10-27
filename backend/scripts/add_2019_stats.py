"""Write a copy of training-data.zip with a 2019 season added from Sleeper.

The MFL labels have 2019 values but the zip starts in 2020, so 2019 never
became a training row. This adds one, for the zip's own players only (the
same population rule as every other season), with the fields predictor_v5
reads: age, years_exp, fantasy_points and games_played.

- Points: Sleeper's weekly ``pts_ppr`` summed over weeks 1-17. Sleeper's
  ``pts_ppr`` reproduces every 2025 zip total exactly.
- Games: weeks with ``gp`` > 0 and nonzero ``pts_ppr``, which is how the
  zip's weekly ``games_played`` lines up with Sleeper's weekly stats.
- Age and years_exp: the player's earliest zip season minus the gap, since
  the zip moves both up by exactly one a season (3,163 of 3,203 steps).
  A player whose years_exp would be negative is skipped.

training-data.zip itself is left alone: the sealed 2026 protocols hash it.

Usage:
    cd backend
    python -m scripts.add_2019_stats --out data/training-data-2019.zip
"""

import argparse
import json
import time
import urllib.request
import zipfile
from pathlib import Path

SOURCE = Path("data/training-data.zip")
SEASON = 2019
WEEKS = range(1, 18)


def fetch_week(week: int, cache: Path | None) -> dict:
    path = cache / f"sleeper_stats_{SEASON}_{week}.json" if cache else None
    if path and path.exists():
        return json.loads(path.read_text())
    url = f"https://api.sleeper.app/v1/stats/nfl/regular/{SEASON}/{week}"
    with urllib.request.urlopen(url, timeout=30) as resp:
        body = resp.read()
    if path:
        path.write_bytes(body)
    time.sleep(0.3)
    return json.loads(body)


def season_row(player: dict, weeks: list[dict]) -> dict | None:
    first = min(player["seasons"], key=lambda s: s["year"])
    gap = first["year"] - SEASON
    years_exp = first["years_exp"] - gap if first.get("years_exp") is not None else None
    if years_exp is not None and years_exp < 0:
        return None
    pid = player["player_id"]
    weekly = []
    for week, stats in zip(WEEKS, weeks):
        row = stats.get(pid) or {}
        pts = float(row.get("pts_ppr") or 0.0)
        played = 1 if (row.get("gp") or 0) > 0 and pts != 0 else 0
        weekly.append({"week": week, "fantasy_points": round(pts, 2), "games_played": played})
    return {
        "year": SEASON,
        "age": first["age"] - gap,
        "years_exp": years_exp,
        "fantasy_points": round(sum(w["fantasy_points"] for w in weekly), 2),
        "games_played": sum(w["games_played"] for w in weekly),
        "weekly_stats": weekly,
        "source": "sleeper-weekly-pts_ppr",
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--source", default=str(SOURCE))
    parser.add_argument("--out", required=True)
    parser.add_argument("--cache", help="directory for Sleeper responses, reused when present")
    args = parser.parse_args()

    cache = Path(args.cache) if args.cache else None
    if cache:
        cache.mkdir(parents=True, exist_ok=True)
    weeks = [fetch_week(w, cache) for w in WEEKS]

    with zipfile.ZipFile(args.source) as zf:
        name = next(n for n in zf.namelist() if n.endswith(".json"))
        data = json.loads(zf.read(name))

    added = skipped = 0
    for player in data["players"]:
        if not player.get("seasons") or any(s["year"] == SEASON for s in player["seasons"]):
            continue
        row = season_row(player, weeks)
        if row is None:
            skipped += 1
            continue
        player["seasons"].insert(0, row)
        added += 1

    with zipfile.ZipFile(args.out, "w", zipfile.ZIP_DEFLATED) as zf:
        zf.writestr(name, json.dumps(data))
    print(f"added {added} {SEASON} seasons, skipped {skipped} with no NFL or college season yet; wrote {args.out}")


if __name__ == "__main__":
    main()
