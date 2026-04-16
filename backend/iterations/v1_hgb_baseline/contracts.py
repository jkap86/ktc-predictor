"""Contract data integration for training and KNN features.

Fetches contract data from nfl_data_py and provides features:
- apy: Annual value of current contract
- apy_cap_pct: APY as percentage of salary cap (market context)
- years_remaining: Years left on contract at season start
- is_contract_year: Whether player is in final year (1 or 0)

These features help distinguish:
1. Players on rookie deals vs. established vets (different upside)
2. Contract year performers (historically elevated production)
3. Elite vs. mid-tier market position
"""

from functools import lru_cache
from typing import Any

import numpy as np
import pandas as pd


@lru_cache(maxsize=1)
def _load_contract_data() -> pd.DataFrame:
    """Load and cache contract data from nfl_data_py.

    Returns
    -------
    pd.DataFrame
        Contract data with gsis_id as key.
    """
    import nfl_data_py as nfl

    contracts = nfl.import_contracts()
    # Keep only relevant columns and positions
    contracts = contracts[contracts["position"].isin(["QB", "RB", "WR", "TE"])]
    return contracts[
        ["gsis_id", "year_signed", "years", "value", "apy", "guaranteed", "apy_cap_pct"]
    ].copy()


@lru_cache(maxsize=1)
def _load_id_mapping() -> dict[str, str]:
    """Load Sleeper ID to GSIS ID mapping.

    Returns
    -------
    dict[str, str]
        Mapping of sleeper_id (string) -> gsis_id (string).
    """
    import nfl_data_py as nfl

    ids = nfl.import_ids()
    # Filter to valid IDs
    valid = ids[ids["sleeper_id"].notna() & ids["gsis_id"].notna()].copy()
    # Convert sleeper_id to string (it's stored as float)
    valid["sleeper_id_str"] = valid["sleeper_id"].astype(int).astype(str)
    return dict(zip(valid["sleeper_id_str"], valid["gsis_id"]))


def get_contract_features(
    sleeper_id: str, season_year: int, position: str
) -> dict[str, Any] | None:
    """Get contract features for a player-season.

    Parameters
    ----------
    sleeper_id : str
        Player's Sleeper ID (e.g., "19", "4046").
    season_year : int
        Season year (e.g., 2024).
    position : str
        Player position (QB, RB, WR, TE).

    Returns
    -------
    dict or None
        Contract features dict, or None if no contract found:
        - apy: Annual contract value in millions
        - apy_cap_pct: APY as % of salary cap
        - contract_years_remaining: Years left on contract
        - is_contract_year: 1 if final year, 0 otherwise
        - apy_position_rank: Percentile rank of APY within position (0-1)
    """
    # Get GSIS ID from Sleeper ID
    id_map = _load_id_mapping()
    gsis_id = id_map.get(sleeper_id)
    if not gsis_id:
        return None

    # Get contracts for this player
    contracts = _load_contract_data()
    player_contracts = contracts[contracts["gsis_id"] == gsis_id]

    if player_contracts.empty:
        return None

    # Find the active contract for this season
    # Contract is active if: year_signed <= season_year < year_signed + years
    active_contracts = player_contracts[
        (player_contracts["year_signed"] <= season_year)
        & (player_contracts["year_signed"] + player_contracts["years"] > season_year)
    ]

    if active_contracts.empty:
        return None

    # Take the most recent contract (in case of multiple)
    contract = active_contracts.sort_values("year_signed", ascending=False).iloc[0]

    year_signed = int(contract["year_signed"]) if pd.notna(contract["year_signed"]) else 0
    years_total = float(contract["years"]) if pd.notna(contract["years"]) else 0

    # Years remaining on contract at start of season
    years_elapsed = season_year - year_signed
    years_remaining = max(0, years_total - years_elapsed)

    # APY and cap percentage
    apy = float(contract["apy"]) if pd.notna(contract["apy"]) else 0.0
    apy_cap_pct = float(contract["apy_cap_pct"]) if pd.notna(contract["apy_cap_pct"]) else 0.0

    # Compute APY position rank (percentile among active contracts for position)
    pos_contracts = contracts[
        (contracts["year_signed"] <= season_year)
        & (contracts["year_signed"] + contracts["years"] > season_year)
    ]
    pos_apys = pos_contracts["apy"].dropna()
    if len(pos_apys) > 0 and apy > 0:
        apy_rank = (pos_apys < apy).sum() / len(pos_apys)
    else:
        apy_rank = 0.5  # Default to median if no data

    return {
        "apy": round(apy, 2),
        "apy_cap_pct": round(apy_cap_pct, 4),
        "contract_years_remaining": round(years_remaining, 1),
        "is_contract_year": 1 if years_remaining <= 1 else 0,
        "apy_position_rank": round(apy_rank, 4),
    }


def compute_contract_features_for_training(
    players: list[dict],
) -> dict[tuple[str, int], dict]:
    """Compute contract features for all player-seasons in training data.

    Parameters
    ----------
    players : list[dict]
        List of player dicts from training data JSON.

    Returns
    -------
    dict[tuple[str, int], dict]
        Mapping of (player_id, year) -> contract features dict.
    """
    result = {}

    # Build lookup structures once
    id_map = _load_id_mapping()
    contracts = _load_contract_data()

    # Group contracts by gsis_id for faster lookup
    contracts_by_gsis = {}
    for _, row in contracts.iterrows():
        gsis_id = row["gsis_id"]
        if gsis_id not in contracts_by_gsis:
            contracts_by_gsis[gsis_id] = []
        contracts_by_gsis[gsis_id].append(row)

    # Compute position-specific APY percentiles per year
    # This gives us relative contract position within the market
    pos_apy_percentiles = {}  # (year, position) -> array of APYs

    for player in players:
        pid = player["player_id"]
        position = player["position"]

        # Map to GSIS ID
        gsis_id = id_map.get(pid)
        if not gsis_id:
            continue

        player_contracts = contracts_by_gsis.get(gsis_id, [])
        if not player_contracts:
            continue

        for season in player.get("seasons", []):
            year = season["year"]

            # Find active contract for this season
            active = None
            for c in player_contracts:
                yr_signed = c["year_signed"] if pd.notna(c["year_signed"]) else 0
                yrs = c["years"] if pd.notna(c["years"]) else 0
                if yr_signed <= year < yr_signed + yrs:
                    if active is None or c["year_signed"] > active["year_signed"]:
                        active = c

            if active is None:
                continue

            year_signed = int(active["year_signed"]) if pd.notna(active["year_signed"]) else 0
            years_total = float(active["years"]) if pd.notna(active["years"]) else 0
            years_remaining = max(0, years_total - (year - year_signed))

            apy = float(active["apy"]) if pd.notna(active["apy"]) else 0.0
            apy_cap_pct = float(active["apy_cap_pct"]) if pd.notna(active["apy_cap_pct"]) else 0.0

            result[(pid, year)] = {
                "apy": round(apy, 2),
                "apy_cap_pct": round(apy_cap_pct, 4),
                "contract_years_remaining": round(years_remaining, 1),
                "is_contract_year": 1 if years_remaining <= 1 else 0,
            }

    # Second pass: compute position percentile ranks
    # Group all APYs by (year, position) from the result
    year_pos_apys: dict[tuple[int, str], list[float]] = {}
    for player in players:
        pid = player["player_id"]
        position = player["position"]
        for season in player.get("seasons", []):
            year = season["year"]
            key = (pid, year)
            if key in result:
                apy = result[key]["apy"]
                if apy > 0:
                    yp_key = (year, position)
                    if yp_key not in year_pos_apys:
                        year_pos_apys[yp_key] = []
                    year_pos_apys[yp_key].append(apy)

    # Compute percentile ranks
    for player in players:
        pid = player["player_id"]
        position = player["position"]
        for season in player.get("seasons", []):
            year = season["year"]
            key = (pid, year)
            if key in result:
                apy = result[key]["apy"]
                yp_key = (year, position)
                apys = year_pos_apys.get(yp_key, [])
                if apys and apy > 0:
                    rank = sum(1 for a in apys if a < apy) / len(apys)
                else:
                    rank = 0.5
                result[key]["apy_position_rank"] = round(rank, 4)

    return result


def get_knn_contract_features(
    sleeper_id: str, season_year: int, position: str
) -> tuple[float, float] | None:
    """Get contract features for KNN similarity matching.

    Returns a subset of features suitable for KNN neighbor matching.
    These features capture the player's market position and contract situation.

    Parameters
    ----------
    sleeper_id : str
        Player's Sleeper ID.
    season_year : int
        Season year.
    position : str
        Player position.

    Returns
    -------
    tuple[float, float] or None
        (apy_cap_pct, is_contract_year) or None if no contract found.
    """
    features = get_contract_features(sleeper_id, season_year, position)
    if features is None:
        return None
    return (features["apy_cap_pct"], features["is_contract_year"])
