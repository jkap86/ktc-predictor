"""Populate KTC market movement features from daily ktc_dynasty DB data.

For each player-season, computes pre-season market features from daily KTC history:
  - ktc_7d_change: 7-day log change before season start
  - ktc_14d_change: 14-day log change
  - ktc_30d_change: 30-day log change (offseason momentum)
  - ktc_60d_change: 60-day log change
  - ktc_90d_change: 90-day log change (full offseason trajectory)
  - ktc_range_90d: (max - min) / mean over 90 days (normalized volatility)
  - ktc_days_since_peak: days since career peak KTC as of season start
  - preseason_ktc_slope: linear regression slope of log(KTC) over 60-day offseason window
  - position_rank_at_start: position rank at season start (from DB)

All features use data strictly before season start (no leakage).

Usage:
    cd backend
    python -m scripts.populate_ktc_market_features
    python -m scripts.populate_ktc_market_features --dry-run
"""

import argparse
import asyncio
import json
import os
import zipfile
from datetime import date, timedelta
from pathlib import Path

import asyncpg
import numpy as np
from dotenv import load_dotenv

load_dotenv()

POSITIONS = {"QB", "RB", "WR", "TE"}

# Season start dates (approximate — before week 1 of NFL season)
SEASON_START_DATES = {
    2020: date(2020, 9, 10),
    2021: date(2021, 9, 9),
    2022: date(2022, 9, 8),
    2023: date(2023, 9, 7),
    2024: date(2024, 9, 5),
    2025: date(2025, 9, 4),
}

# How many days of history to fetch before season start
LOOKBACK_DAYS = 365


async def fetch_ktc_history(
    conn, player_id: str, before_date: date, days_back: int = LOOKBACK_DAYS
) -> list[dict]:
    """Fetch daily KTC records for a player in the window [before_date - days_back, before_date)."""
    start_date = before_date - timedelta(days=days_back)
    rows = await conn.fetch(
        """
        SELECT date, value, position_rank
        FROM ktc_dynasty
        WHERE player_id = $1 AND date >= $2 AND date < $3
        ORDER BY date ASC
        """,
        player_id,
        start_date,
        before_date,
    )
    return [{"date": r["date"], "value": float(r["value"]), "position_rank": r["position_rank"]} for r in rows]


def compute_market_features(history: list[dict], season_start: date) -> dict:
    """Compute market movement features from daily KTC history.

    All features are computed using data strictly before season_start.
    """
    if not history or len(history) < 3:
        return {}

    dates = [h["date"] for h in history]
    values = np.array([h["value"] for h in history])

    # Latest value and position rank
    latest = history[-1]
    latest_value = latest["value"]
    position_rank = latest["position_rank"]

    features = {}

    if position_rank is not None:
        features["position_rank_at_start"] = int(position_rank)

    # Log-change features over various windows
    for label, days in [("7d", 7), ("14d", 14), ("30d", 30), ("60d", 60), ("90d", 90)]:
        cutoff = season_start - timedelta(days=days)
        # Find the value closest to cutoff date
        past_vals = [(h["date"], h["value"]) for h in history if h["date"] <= cutoff and h["value"] > 0]
        if past_vals and latest_value > 0:
            past_value = past_vals[-1][1]  # most recent before cutoff
            features[f"ktc_{label}_change"] = round(
                float(np.log(latest_value / past_value)), 4
            )

    # 90-day normalized range (volatility proxy)
    cutoff_90 = season_start - timedelta(days=90)
    recent_vals = np.array([h["value"] for h in history if h["date"] >= cutoff_90])
    if len(recent_vals) >= 5:
        mean_val = recent_vals.mean()
        if mean_val > 0:
            features["ktc_range_90d"] = round(
                float((recent_vals.max() - recent_vals.min()) / mean_val), 4
            )

    # Days since career peak
    peak_idx = int(np.argmax(values))
    peak_date = dates[peak_idx]
    days_since_peak = (season_start - peak_date).days
    features["ktc_days_since_peak"] = days_since_peak

    # Preseason slope: linear regression of log(KTC) over 60-day offseason window
    cutoff_60 = season_start - timedelta(days=60)
    window_data = [(h["date"], h["value"]) for h in history if h["date"] >= cutoff_60 and h["value"] > 0]
    if len(window_data) >= 5:
        x = np.array([(d - cutoff_60).days for d, v in window_data], dtype=float)
        y = np.log(np.array([v for d, v in window_data]))
        if np.std(x) > 0:
            slope = float(np.polyfit(x, y, 1)[0])
            # Slope is in log-KTC per day — multiply by 30 for monthly rate
            features["preseason_ktc_slope"] = round(slope * 30, 4)

    return features


async def main_async(data_path: str, dry_run: bool):
    with open(data_path) as f:
        data = json.load(f)

    db_url = os.getenv("DATABASE_URL")
    if not db_url:
        print("ERROR: DATABASE_URL not set. Create .env or set environment variable.")
        return

    conn = await asyncpg.connect(db_url)

    # Check DB date range
    row = await conn.fetchrow("SELECT MIN(date) AS min_date, MAX(date) AS max_date, COUNT(*) AS n FROM ktc_dynasty")
    print(f"Database: {row['n']} records, {row['min_date']} to {row['max_date']}")
    print()

    filled_counts = {}
    total_seasons = 0
    skipped_no_history = 0

    for player in data["players"]:
        pid = str(player["player_id"])
        pos = player["position"]
        if pos not in POSITIONS:
            continue

        for season in player.get("seasons", []):
            year = season["year"]
            if year not in SEASON_START_DATES:
                continue
            total_seasons += 1

            season_start = SEASON_START_DATES[year]
            history = await fetch_ktc_history(conn, pid, season_start)

            if not history:
                skipped_no_history += 1
                continue

            features = compute_market_features(history, season_start)

            for key, value in features.items():
                season[key] = value
                filled_counts[key] = filled_counts.get(key, 0) + 1

    await conn.close()

    print(f"Processed {total_seasons} player-seasons")
    print(f"Skipped (no DB history): {skipped_no_history}")
    print()
    print("Features populated:")
    for key in sorted(filled_counts.keys()):
        count = filled_counts[key]
        pct = 100 * count / total_seasons if total_seasons else 0
        print(f"  {key:<30} {count:>5} ({pct:>5.1f}%)")

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
    parser = argparse.ArgumentParser(
        description="Populate KTC market features from daily DB data"
    )
    parser.add_argument("--data", default="data/training-data.json")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    asyncio.run(main_async(args.data, args.dry_run))


if __name__ == "__main__":
    main()
