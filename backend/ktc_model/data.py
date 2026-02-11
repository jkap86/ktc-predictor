"""Build weekly cumulative snapshot DataFrame from training data."""

import json
import zipfile
from pathlib import Path

import numpy as np
import pandas as pd

from .contracts import compute_contract_features_for_training
from .pff_grades import compute_pff_features_for_training

VALID_POSITIONS = {"QB", "RB", "WR", "TE"}

# Prime age by position (peak dynasty value age)
PRIME_AGE = {"QB": 27, "RB": 24, "WR": 26, "TE": 27}

# Position-specific PPG baselines for z-score calculation
PPG_BASELINES = {
    "QB": {"mean": 18.0, "std": 5.0},
    "RB": {"mean": 12.0, "std": 5.0},
    "WR": {"mean": 10.0, "std": 4.0},
    "TE": {"mean": 8.0, "std": 3.5},
}

# KTC quartile boundaries (from analysis)
KTC_QUARTILE_BOUNDS = [1559, 3085, 4850]


def _get_ktc_quartile(ktc: float) -> int:
    """Return KTC quartile (1-4) based on fixed boundaries."""
    if ktc < 1559:
        return 1
    elif ktc < 3085:
        return 2
    elif ktc < 4850:
        return 3
    return 4


def _age_prime_distance(age: float | None, position: str) -> float:
    """Return age distance from positional prime. Negative = before prime."""
    if age is None:
        return 0.0
    prime = PRIME_AGE.get(position, 26)
    return age - prime


def _ppg_zscore(ppg: float, position: str) -> float:
    """Return PPG as z-score relative to position average."""
    baseline = PPG_BASELINES.get(position, {"mean": 12.0, "std": 5.0})
    return (ppg - baseline["mean"]) / baseline["std"]


def _is_breakout_candidate(
    age: float | None, ktc: float, ppg: float, position: str
) -> int:
    """Binary flag: 1 if player matches breakout profile, 0 otherwise.

    Breakout profile:
    - Mid-tier KTC (Q2-Q3)
    - Prime age window (within 3 years of prime)
    - Above-average PPG (z-score > 0.5)
    """
    if age is None:
        return 0

    # Check KTC tier
    ktc_q = _get_ktc_quartile(ktc)
    if ktc_q not in (2, 3):
        return 0

    # Check age (within 3 years of prime)
    prime = PRIME_AGE.get(position, 26)
    if abs(age - prime) > 3:
        return 0

    # Check PPG (above average for position)
    zscore = _ppg_zscore(ppg, position)
    if zscore < 0.5:
        return 0

    return 1


def _compute_prior_season_features(players: list[dict]) -> dict[tuple[str, int], dict]:
    """Compute prior-year KTC features for all player-seasons.

    For each (player_id, year), computes features based on the PRIOR season's
    end_ktc value. This captures player trajectory ("path") that start_ktc alone
    misses - distinguishing risers from fallers in the same tier.

    Parameters
    ----------
    players : list[dict]
        List of player dicts from training data.

    Returns
    -------
    dict[tuple[str, int], dict]
        Mapping of (player_id, year) -> {
            'prior_end_ktc': float,       # Prior season's end_ktc
            'max_ktc_prior': float,       # Career max KTC up to prior season
            'ktc_yoy_log': float,         # log(start_ktc / prior_end_ktc), clipped
            'ktc_peak_drawdown': float,   # log(start_ktc / max_ktc_prior)
        }
        Only includes entries for seasons with valid prior-year data.
    """
    result = {}

    for player in players:
        pid = player["player_id"]
        seasons = player.get("seasons", [])

        if len(seasons) < 2:
            continue

        # Sort seasons by year, excluding pre-draft (college) data
        sorted_seasons = sorted(
            [s for s in seasons if (s.get("years_exp") or 0) >= 0],
            key=lambda s: s["year"],
        )

        # Track running max of end_ktc values seen so far
        max_ktc_so_far = 0.0

        for i, season in enumerate(sorted_seasons):
            year = season["year"]
            start_ktc = season.get("start_ktc")

            if i == 0:
                # First season: no prior data, but track for future seasons
                end_ktc = season.get("end_ktc")
                if end_ktc and end_ktc > 0 and end_ktc < 9999:
                    max_ktc_so_far = max(max_ktc_so_far, end_ktc)
                continue

            # Get prior season's end_ktc
            prior_season = sorted_seasons[i - 1]
            prior_end_ktc = prior_season.get("end_ktc")

            # Skip if missing or invalid values
            if not start_ktc or start_ktc <= 0 or start_ktc >= 9999:
                # Still update max for next iteration
                end_ktc = season.get("end_ktc")
                if end_ktc and end_ktc > 0 and end_ktc < 9999:
                    max_ktc_so_far = max(max_ktc_so_far, end_ktc)
                continue

            if not prior_end_ktc or prior_end_ktc <= 0 or prior_end_ktc >= 9999:
                # Still update max for next iteration
                end_ktc = season.get("end_ktc")
                if end_ktc and end_ktc > 0 and end_ktc < 9999:
                    max_ktc_so_far = max(max_ktc_so_far, end_ktc)
                continue

            # Compute YoY log ratio (clipped to avoid outliers)
            # Positive = rose into tier, Negative = fell into tier
            ktc_yoy_log = float(np.log(start_ktc / prior_end_ktc))
            ktc_yoy_log = float(np.clip(ktc_yoy_log, -0.7, 0.7))

            # Compute peak drawdown (using max from all prior seasons)
            # Update max to include prior season before computing
            max_ktc_prior = max(max_ktc_so_far, prior_end_ktc)

            if max_ktc_prior > 0:
                # 0 = at career peak, Negative = below peak
                ktc_peak_drawdown = float(np.log(start_ktc / max_ktc_prior))
            else:
                ktc_peak_drawdown = 0.0

            result[(pid, year)] = {
                "prior_end_ktc": round(prior_end_ktc, 1),
                "max_ktc_prior": round(max_ktc_prior, 1),
                "ktc_yoy_log": round(ktc_yoy_log, 4),
                "ktc_peak_drawdown": round(ktc_peak_drawdown, 4),
            }

            # Update running max for next iteration
            max_ktc_so_far = max_ktc_prior
            end_ktc = season.get("end_ktc")
            if end_ktc and end_ktc > 0 and end_ktc < 9999:
                max_ktc_so_far = max(max_ktc_so_far, end_ktc)

    return result


def _compute_prior_ppg_features(players: list[dict]) -> dict[tuple[str, int], dict]:
    """Compute prior-year PPG features for all player-seasons.

    For each (player_id, year), computes features based on the PRIOR season's
    PPG. This captures performance trajectory that complements KTC trajectory.

    Parameters
    ----------
    players : list[dict]
        List of player dicts from training data.

    Returns
    -------
    dict[tuple[str, int], dict]
        Mapping of (player_id, year) -> {
            'prior_ppg': float,       # Prior season's PPG
            'ppg_yoy_log': float,     # log(ppg_so_far / prior_ppg), clipped
            'has_prior_ppg': int,     # 1 if valid prior PPG exists
        }
        Only includes entries for seasons with valid prior-year PPG data.
    """
    result = {}

    for player in players:
        pid = player["player_id"]
        seasons = player.get("seasons", [])

        if len(seasons) < 2:
            continue

        # Sort seasons by year to ensure correct ordering
        sorted_seasons = sorted(seasons, key=lambda s: s["year"])

        for i, season in enumerate(sorted_seasons):
            if i == 0:
                continue  # No prior data for first season

            year = season["year"]
            prior_season = sorted_seasons[i - 1]

            # Get prior season PPG
            prior_games = prior_season.get("games_played", 0) or 0
            prior_fp = prior_season.get("fantasy_points", 0) or 0

            if prior_games < 4:  # Require meaningful sample
                continue

            prior_ppg = prior_fp / prior_games

            # Get current season PPG (full season for training)
            curr_games = season.get("games_played", 0) or 0
            curr_fp = season.get("fantasy_points", 0) or 0

            if curr_games < 1:
                continue

            curr_ppg = curr_fp / curr_games

            # Log ratio with epsilon to avoid log(0)
            eps = 0.1
            ppg_yoy_log = float(np.log((curr_ppg + eps) / (prior_ppg + eps)))
            ppg_yoy_log = float(np.clip(ppg_yoy_log, -1.0, 1.0))  # Clip outliers

            result[(pid, year)] = {
                "prior_ppg": round(prior_ppg, 2),
                "ppg_yoy_log": round(ppg_yoy_log, 4),
                "has_prior_ppg": 1,
            }

    return result


def _compute_offseason_features(players: list[dict]) -> dict[tuple[str, int], dict]:
    """Compute offseason features for all player-seasons.

    For each (player_id, year), computes:
    - offseason_percentile: where start_ktc sits in offseason range [0=low, 1=high]
    - trend_14d: log(last_ktc / ktc_14d_ago) - late offseason momentum

    This captures two signals:
    1. Mean reversion: players at offseason peak tend to regress
    2. Momentum: players trending up in late offseason may continue rising

    Parameters
    ----------
    players : list[dict]
        List of player dicts from training data.

    Returns
    -------
    dict[tuple[str, int], dict]
        Mapping of (player_id, year) -> {
            'offseason_percentile': float (0.0 to 1.0),
            'trend_14d': float (log ratio, typically -0.5 to +0.5),
        }
        Only includes entries with sufficient offseason KTC data.
    """
    result = {}

    for player in players:
        pid = player["player_id"]

        for season in player.get("seasons", []):
            # Skip pre-draft seasons (college data)
            if (season.get("years_exp") or 0) < 0:
                continue

            offseason_ktc = season.get("offseason_ktc", [])
            start_ktc = season.get("start_ktc")

            if not offseason_ktc or not start_ktc or start_ktc <= 0:
                continue

            # Sort by date and filter valid (non-zero) entries
            sorted_entries = sorted(
                [e for e in offseason_ktc if e.get("ktc", 0) > 0],
                key=lambda x: x["date"],
            )

            if len(sorted_entries) < 14:  # Need enough data points
                continue

            ktc_values = [e["ktc"] for e in sorted_entries]
            offseason_high = max(ktc_values)
            offseason_low = min(ktc_values)
            offseason_range = offseason_high - offseason_low

            # Offseason percentile
            if offseason_range > 0:
                # 0 = at low (room to grow), 1 = at high (likely to regress)
                percentile = (start_ktc - offseason_low) / offseason_range
                percentile = max(0.0, min(1.0, percentile))
            else:
                percentile = 0.5  # No range = neutral

            # Trend 14d: log(last / 14d_ago)
            # Captures late-offseason momentum heading into the season
            last_ktc = sorted_entries[-1]["ktc"]
            ktc_14d_ago = sorted_entries[-14]["ktc"]
            if ktc_14d_ago > 0:
                trend_14d = float(np.log(last_ktc / ktc_14d_ago))
            else:
                trend_14d = 0.0

            result[(pid, season["year"])] = {
                "offseason_percentile": round(percentile, 4),
                "trend_14d": round(trend_14d, 4),
            }

    return result


def build_weekly_snapshot_df(
    zip_path: str, json_name: str = "training-data.json"
) -> pd.DataFrame:
    """Load training data and produce weekly cumulative snapshots.

    For each player-season, walks through weekly_stats in week order,
    accumulating games_played and fantasy_points to compute a running ppg.

    Parameters
    ----------
    zip_path : str
        Path to the zip file containing the training JSON.
    json_name : str
        Name of the JSON file inside the zip.

    Returns
    -------
    pd.DataFrame
        Columns: player_id, position, year, week, games_played_so_far,
                 ppg_so_far, start_ktc, age, weeks_missed_so_far,
                 draft_pick, years_remaining, end_ktc, abs_change,
                 offseason_percentile
    """
    zip_path = Path(zip_path)
    with zipfile.ZipFile(zip_path, "r") as zf:
        with zf.open(json_name) as f:
            data = json.load(f)

    # Pre-compute features for all player-seasons
    offseason_feats = _compute_offseason_features(data["players"])
    prior_season_feats = _compute_prior_season_features(data["players"])
    prior_ppg_feats = _compute_prior_ppg_features(data["players"])

    # Contract features from nfl_data_py (apy, apy_cap_pct, years_remaining, is_contract_year)
    try:
        contract_feats = compute_contract_features_for_training(data["players"])
        print(f"  Contract features loaded for {len(contract_feats)} player-seasons")
    except Exception as e:
        print(f"  Warning: Could not load contract features: {e}")
        contract_feats = {}

    # PFF grades (efficiency beyond PPG, especially helpful for RB)
    try:
        pff_feats = compute_pff_features_for_training(data["players"])
        print(f"  PFF grades loaded for {len(pff_feats)} player-seasons")
    except Exception as e:
        print(f"  Warning: Could not load PFF grades: {e}")
        pff_feats = {}

    rows: list[dict] = []

    for player in data["players"]:
        pid = player["player_id"]
        position = player["position"]

        if position not in VALID_POSITIONS:
            continue

        for season in player["seasons"]:
            # Skip pre-draft seasons (college data)
            if (season.get("years_exp") or 0) < 0:
                continue

            end_ktc = season.get("end_ktc")
            if end_ktc is None:
                continue

            start_ktc = season.get("start_ktc")
            if start_ktc is None or start_ktc <= 0:
                continue

            year = season["year"]
            age = season.get("age")
            draft_pick = season.get("draft_pick")
            years_remaining = season.get("years_remaining")
            weekly_stats = season.get("weekly_stats", [])

            # Team context features (from training data)
            qb_ktc = season.get("qb_ktc")
            team_total_ktc = season.get("team_total_ktc")
            positional_competition = season.get("positional_competition")

            # Sort by week to ensure correct accumulation
            weekly_stats = sorted(weekly_stats, key=lambda w: w["week"])

            games_so_far = 0
            fp_so_far = 0.0

            for ws in weekly_stats:
                games_so_far += ws.get("games_played", 0)
                fp_so_far += ws.get("fantasy_points", 0.0)

                if games_so_far <= 0:
                    continue

                ppg_so_far = fp_so_far / games_so_far
                weeks_missed_so_far = ws["week"] - games_so_far

                # Look up precomputed features for this player-season
                off_feat = offseason_feats.get((pid, year), {})
                prior_feat = prior_season_feats.get((pid, year), {})
                ppg_feat = prior_ppg_feats.get((pid, year), {})
                contract_feat = contract_feats.get((pid, year), {})
                pff_feat = pff_feats.get((pid, year), {})

                rows.append(
                    {
                        "player_id": pid,
                        "position": position,
                        "year": year,
                        "week": ws["week"],
                        "games_played_so_far": games_so_far,
                        "ppg_so_far": round(ppg_so_far, 4),
                        "start_ktc": start_ktc,
                        "age": age,
                        "weeks_missed_so_far": weeks_missed_so_far,
                        "draft_pick": draft_pick,
                        "years_remaining": years_remaining,
                        "end_ktc": end_ktc,
                        "abs_change": end_ktc - start_ktc,
                        "log_ratio": np.log(max(end_ktc, 1) / start_ktc),
                        "pct_change": (end_ktc - start_ktc) / start_ktc,
                        # Engineered features
                        "start_ktc_quartile": _get_ktc_quartile(start_ktc),
                        "age_prime_distance": _age_prime_distance(age, position),
                        "ppg_zscore": round(_ppg_zscore(ppg_so_far, position), 4),
                        "is_breakout_candidate": _is_breakout_candidate(
                            age, start_ktc, ppg_so_far, position
                        ),
                        # Offseason features (disabled in model - USE_OFFSEASON_FEATURES=False)
                        "offseason_percentile": off_feat.get("offseason_percentile"),
                        "trend_14d": off_feat.get("trend_14d"),
                        # Prior-season KTC features (captures trajectory/path)
                        # ktc_yoy_log: log(start_ktc / prior_end_ktc), clipped to [-0.7, 0.7]
                        # Positive = rose into tier, Negative = fell into tier
                        "ktc_yoy_log": prior_feat.get("ktc_yoy_log"),
                        # ktc_peak_drawdown: log(start_ktc / max_ktc_prior)
                        # 0 = at career peak, Negative = below peak
                        "ktc_peak_drawdown": prior_feat.get("ktc_peak_drawdown"),
                        # has_prior_season: 1 if prior season data exists, else 0
                        "has_prior_season": 1 if prior_feat else 0,
                        # Prior-season PPG features (QB only - captures performance trajectory)
                        # prior_ppg: prior season's PPG (absolute baseline)
                        "prior_ppg": ppg_feat.get("prior_ppg"),
                        # ppg_yoy_log: log(ppg_so_far / prior_ppg), clipped to [-1.0, 1.0]
                        # Positive = performing better, Negative = performing worse
                        "ppg_yoy_log": ppg_feat.get("ppg_yoy_log"),
                        # has_prior_ppg: 1 if valid prior PPG exists, else 0
                        "has_prior_ppg": 1 if ppg_feat else 0,
                        # Contract features (from nfl_data_py)
                        # apy_cap_pct: APY as % of salary cap (0.0 to ~0.25)
                        "apy_cap_pct": contract_feat.get("apy_cap_pct"),
                        # is_contract_year: 1 if in final year, 0 otherwise
                        "is_contract_year": contract_feat.get("is_contract_year"),
                        # apy_position_rank: percentile of APY within position (0-1)
                        "apy_position_rank": contract_feat.get("apy_position_rank"),
                        # has_contract_data: 1 if contract info exists, else 0
                        "has_contract_data": 1 if contract_feat else 0,
                        # PFF grades (efficiency beyond PPG)
                        # pff_overall_grade: 0-100 composite rating
                        "pff_overall_grade": pff_feat.get("pff_overall_grade"),
                        # pff_grade_tier: 0=reserve, 1=starter, 2=elite
                        "pff_grade_tier": pff_feat.get("pff_grade_tier"),
                        # pff_position_grade: position-specific grade (run/receiving)
                        "pff_position_grade": pff_feat.get("pff_position_grade"),
                        # has_pff_data: 1 if PFF data exists, else 0
                        "has_pff_data": 1 if pff_feat else 0,
                        # Team context features
                        # qb_ktc: Team's QB KTC value (opportunity quality)
                        "qb_ktc": qb_ktc,
                        # team_total_ktc: Sum of all teammates' KTC (roster strength)
                        "team_total_ktc": team_total_ktc,
                        # positional_competition: KTC of same-position teammates (RB committee risk)
                        "positional_competition": positional_competition,
                    }
                )

    df = pd.DataFrame(rows)

    # Mark and replace 9999 sentinels in both start_ktc and end_ktc
    df["start_ktc_was_sentinel"] = 0
    for pos in VALID_POSITIONS:
        mask = df["position"] == pos
        if not mask.any():
            continue
        # start_ktc sentinels → median (conservative; flag compensates)
        start_sent = mask & (df["start_ktc"] >= 9999)
        if start_sent.any():
            non_sent = mask & (df["start_ktc"] < 9999)
            df.loc[start_sent, "start_ktc_was_sentinel"] = 1
            df.loc[start_sent, "start_ktc"] = df.loc[non_sent, "start_ktc"].median()
        # end_ktc sentinels → p95 (closer to truth for target)
        end_sent = mask & (df["end_ktc"] >= 9999)
        if end_sent.any():
            non_sent_end = mask & (df["end_ktc"] < 9999)
            df.loc[end_sent, "end_ktc"] = df.loc[non_sent_end, "end_ktc"].quantile(0.95)
        # Recompute derived columns for entire position
        df.loc[mask, "log_ratio"] = np.log(
            np.maximum(df.loc[mask, "end_ktc"], 1) / df.loc[mask, "start_ktc"]
        )
        df.loc[mask, "abs_change"] = df.loc[mask, "end_ktc"] - df.loc[mask, "start_ktc"]
        df.loc[mask, "pct_change"] = (
            (df.loc[mask, "end_ktc"] - df.loc[mask, "start_ktc"]) / df.loc[mask, "start_ktc"]
        )

    return df
