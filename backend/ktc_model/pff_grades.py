"""PFF (Pro Football Focus) grades integration for training features.

Loads PFF grades from backend/data/pff-grades-*.json files and provides
features for model training. PFF grades capture player efficiency beyond
raw fantasy points, which helps distinguish sustainable elite performance
from fluky production.

Available grades:
- overall_grade: 0-100 composite rating
- pass_grade: Passing efficiency (QB)
- run_grade: Rushing efficiency (RB, QB scrambles)
- receiving_grade: Receiving efficiency (WR, RB, TE)
- pass_block_grade: Pass blocking (OL, TE)
- run_block_grade: Run blocking (OL, TE)
"""

import json
from functools import lru_cache
from pathlib import Path
from typing import Any


# PFF grade tier boundaries
PFF_TIER_ELITE = 80  # Top performers
PFF_TIER_STARTER = 70  # Solid starters
# Below 70 = reserve/replacement level


@lru_cache(maxsize=1)
def _load_all_pff_grades() -> dict[tuple[str, int], dict]:
    """Load PFF grades from all available years.

    Returns
    -------
    dict[tuple[str, int], dict]
        Mapping of (player_id, season) -> grade data dict.
    """
    data_dir = Path(__file__).parent.parent / "data"
    result = {}

    for year in range(2021, 2026):  # 2021-2025
        pff_file = data_dir / f"pff-grades-{year}.json"
        if not pff_file.exists():
            continue

        with open(pff_file) as f:
            grades = json.load(f)

        for g in grades:
            player_id = g.get("player_id")
            season = g.get("season", year)
            if player_id:
                result[(player_id, season)] = g

    return result


def get_pff_features(player_id: str, season: int, position: str) -> dict[str, Any] | None:
    """Get PFF features for a player-season.

    Parameters
    ----------
    player_id : str
        Player's Sleeper ID.
    season : int
        Season year (e.g., 2024).
    position : str
        Player position (QB, RB, WR, TE).

    Returns
    -------
    dict or None
        PFF features dict, or None if no PFF data found:
        - pff_overall_grade: 0-100 composite rating
        - pff_grade_tier: 0=reserve, 1=starter, 2=elite
        - pff_position_grade: Position-specific grade (run/receiving)
        - has_pff_data: 1 if data exists, 0 otherwise
    """
    all_grades = _load_all_pff_grades()
    grade_data = all_grades.get((player_id, season))

    if not grade_data:
        return None

    overall = grade_data.get("overall_grade")
    if overall is None:
        return None

    # Determine grade tier
    if overall >= PFF_TIER_ELITE:
        tier = 2
    elif overall >= PFF_TIER_STARTER:
        tier = 1
    else:
        tier = 0

    # Position-specific grade
    if position == "QB":
        pos_grade = grade_data.get("pass_grade") or overall
    elif position == "RB":
        # RBs: blend of run and receiving grades
        run_g = grade_data.get("run_grade")
        rec_g = grade_data.get("receiving_grade")
        if run_g and rec_g:
            pos_grade = 0.7 * run_g + 0.3 * rec_g  # Weight rushing more
        elif run_g:
            pos_grade = run_g
        elif rec_g:
            pos_grade = rec_g
        else:
            pos_grade = overall
    elif position in ("WR", "TE"):
        pos_grade = grade_data.get("receiving_grade") or overall
    else:
        pos_grade = overall

    return {
        "pff_overall_grade": round(overall, 1),
        "pff_grade_tier": tier,
        "pff_position_grade": round(pos_grade, 1) if pos_grade else None,
        "pff_total_snaps": grade_data.get("total_snaps"),
        "has_pff_data": 1,
    }


def compute_pff_features_for_training(players: list[dict]) -> dict[tuple[str, int], dict]:
    """Compute PFF features for all player-seasons in training data.

    Parameters
    ----------
    players : list[dict]
        List of player dicts from training data JSON.

    Returns
    -------
    dict[tuple[str, int], dict]
        Mapping of (player_id, year) -> PFF features dict.
    """
    result = {}
    all_grades = _load_all_pff_grades()

    for player in players:
        pid = player["player_id"]
        position = player["position"]

        for season in player.get("seasons", []):
            year = season["year"]
            key = (pid, year)

            if key in all_grades:
                features = get_pff_features(pid, year, position)
                if features:
                    result[key] = features

    return result


def get_pff_grade_percentile(grade: float, position: str) -> float:
    """Get percentile rank of a PFF grade within position.

    Uses approximate position-specific distributions based on historical data.

    Parameters
    ----------
    grade : float
        PFF overall grade (0-100).
    position : str
        Player position.

    Returns
    -------
    float
        Percentile rank (0-1).
    """
    # Approximate position medians and stddevs from historical PFF data
    # These are rough estimates; could be computed from actual data
    position_stats = {
        "QB": {"mean": 68, "std": 12},
        "RB": {"mean": 65, "std": 10},
        "WR": {"mean": 63, "std": 11},
        "TE": {"mean": 62, "std": 10},
    }

    stats = position_stats.get(position, {"mean": 65, "std": 10})

    # Z-score to percentile (approximate using normal CDF)
    import math
    z = (grade - stats["mean"]) / stats["std"]
    percentile = 0.5 * (1 + math.erf(z / math.sqrt(2)))

    return round(percentile, 4)
