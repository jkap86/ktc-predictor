"""Prediction and rollout functions for transition models.

Provides single-step prediction (week t → t+1) and full-season rollouts.
"""

import numpy as np

# Position-specific prime ages
PRIME_AGES = {"QB": 27, "RB": 24, "WR": 26, "TE": 27}


def predict_next_ktc(
    model,
    clip_bounds: tuple[float, float],
    ktc_current: float,
    ppg_cumulative: float,
    games_played: int,
    week: int,
    weekly_fp: float = 0.0,
    games_this_week: int = 0,
    ktc_momentum: float | None = None,
    age: float | None = None,
    position: str = "WR",
) -> dict:
    """Predict KTC for next week.

    Parameters
    ----------
    model : sklearn Pipeline
        Trained transition model.
    clip_bounds : tuple[float, float]
        (min, max) bounds for predicted log delta.
    ktc_current : float
        Current KTC value.
    ppg_cumulative : float
        Cumulative PPG through this week.
    games_played : int
        Total games played through this week.
    week : int
        Current week number (1-18).
    weekly_fp : float
        Fantasy points scored this week.
    games_this_week : int
        Games played this week (0 or 1).
    ktc_momentum : float or None
        log(ktc_t / ktc_{t-1}) from previous week.
    age : float or None
        Player age.
    position : str
        Player position.

    Returns
    -------
    dict
        {
            "ktc_next": float,
            "delta_log": float,
            "delta_pct": float,
        }
    """
    prime_age = PRIME_AGES.get(position, 26)
    age_prime_distance = (age - prime_age) if age is not None else np.nan

    season_progress = week / 18.0
    ktc_current_log = np.log(max(ktc_current, 1))

    # Build feature vector
    # Order must match: ktc_current_log, ppg_cumulative, games_played,
    #                   weekly_fp, games_this_week, season_progress,
    #                   ktc_momentum, age_prime_distance
    X = np.array([[
        ktc_current_log,
        ppg_cumulative,
        games_played,
        weekly_fp,
        games_this_week,
        season_progress,
        ktc_momentum if ktc_momentum is not None else np.nan,
        age_prime_distance,
    ]])

    # Predict log delta
    pred_delta_log = float(model.predict(X)[0])

    # Clip to bounds
    pred_delta_log = np.clip(pred_delta_log, clip_bounds[0], clip_bounds[1])

    # Compute next KTC
    ktc_next = ktc_current * np.exp(pred_delta_log)
    ktc_next = max(1.0, min(9999.0, ktc_next))  # Clamp to valid range

    delta_pct = (np.exp(pred_delta_log) - 1) * 100

    return {
        "ktc_next": round(ktc_next, 1),
        "delta_log": round(pred_delta_log, 4),
        "delta_pct": round(delta_pct, 2),
    }


def rollout_season(
    model,
    clip_bounds: tuple[float, float],
    start_ktc: float,
    weekly_stats: list[dict],
    age: float | None = None,
    position: str = "WR",
    start_week: int = 1,
    end_week: int = 18,
) -> list[dict]:
    """Roll forward through a season to generate KTC trajectory.

    Parameters
    ----------
    model : sklearn Pipeline
        Trained transition model.
    clip_bounds : tuple[float, float]
        (min, max) bounds for predicted log delta.
    start_ktc : float
        KTC value at start of rollout.
    weekly_stats : list[dict]
        List of weekly stat dicts: [{"week": 1, "fp": 20.5, "games": 1}, ...]
    age : float or None
        Player age.
    position : str
        Player position.
    start_week : int
        Week to start rollout.
    end_week : int
        Week to end rollout.

    Returns
    -------
    list[dict]
        Trajectory: [{"week": 1, "ktc": 5000}, {"week": 2, "ktc": 5050}, ...]
    """
    # Build stats lookup
    stats_by_week = {}
    for ws in weekly_stats:
        week = ws.get("week")
        if week is not None:
            stats_by_week[week] = {
                "fp": ws.get("fp", ws.get("fantasy_points", 0)) or 0,
                "games": ws.get("games", ws.get("games_played", 0)) or 0,
            }

    trajectory = []
    ktc_current = start_ktc
    ktc_prev = None

    cumulative_fp = 0.0
    cumulative_games = 0

    for week in range(start_week, end_week + 1):
        # Record current state
        trajectory.append({
            "week": week,
            "ktc": round(ktc_current, 1),
        })

        if week == end_week:
            break  # Don't predict beyond end_week

        # Get this week's stats
        ws = stats_by_week.get(week, {"fp": 0, "games": 0})
        weekly_fp = ws["fp"]
        games_this_week = ws["games"]

        # Accumulate stats
        cumulative_fp += weekly_fp
        cumulative_games += games_this_week
        ppg = cumulative_fp / cumulative_games if cumulative_games > 0 else 0.0

        # Compute momentum
        ktc_momentum = None
        if ktc_prev is not None and ktc_prev > 0:
            ktc_momentum = np.log(ktc_current / ktc_prev)

        # Predict next week
        result = predict_next_ktc(
            model=model,
            clip_bounds=clip_bounds,
            ktc_current=ktc_current,
            ppg_cumulative=ppg,
            games_played=cumulative_games,
            week=week,
            weekly_fp=weekly_fp,
            games_this_week=games_this_week,
            ktc_momentum=ktc_momentum,
            age=age,
            position=position,
        )

        # Update state
        ktc_prev = ktc_current
        ktc_current = result["ktc_next"]

    return trajectory


def predict_end_ktc_via_rollout(
    model,
    clip_bounds: tuple[float, float],
    start_ktc: float,
    weekly_stats: list[dict],
    age: float | None = None,
    position: str = "WR",
) -> dict:
    """Predict end-of-season KTC by rolling out weekly predictions.

    This is an alternative to direct EOS prediction, using the transition model.

    Parameters
    ----------
    model : sklearn Pipeline
        Trained transition model.
    clip_bounds : tuple[float, float]
        (min, max) bounds for predicted log delta.
    start_ktc : float
        KTC at start of season.
    weekly_stats : list[dict]
        Weekly stats for the season.
    age : float or None
        Player age.
    position : str
        Player position.

    Returns
    -------
    dict
        {
            "start_ktc": float,
            "end_ktc": float,
            "delta_ktc": float,
            "delta_pct": float,
            "trajectory": list[dict],
        }
    """
    trajectory = rollout_season(
        model=model,
        clip_bounds=clip_bounds,
        start_ktc=start_ktc,
        weekly_stats=weekly_stats,
        age=age,
        position=position,
    )

    end_ktc = trajectory[-1]["ktc"] if trajectory else start_ktc
    delta_ktc = end_ktc - start_ktc
    delta_pct = (delta_ktc / start_ktc * 100) if start_ktc > 0 else 0

    return {
        "start_ktc": round(start_ktc, 1),
        "end_ktc": round(end_ktc, 1),
        "delta_ktc": round(delta_ktc, 1),
        "delta_pct": round(delta_pct, 2),
        "trajectory": trajectory,
    }


def generate_what_if_trajectory(
    model,
    clip_bounds: tuple[float, float],
    start_ktc: float,
    target_ppg: float,
    target_games: int,
    age: float | None = None,
    position: str = "WR",
    current_week: int = 1,
) -> list[dict]:
    """Generate trajectory for a what-if scenario.

    Simulates a player maintaining a target PPG over remaining weeks.

    Parameters
    ----------
    model : sklearn Pipeline
        Trained transition model.
    clip_bounds : tuple[float, float]
        (min, max) bounds for predicted log delta.
    start_ktc : float
        Current KTC value.
    target_ppg : float
        Target PPG for the simulation.
    target_games : int
        Target total games for the season.
    age : float or None
        Player age.
    position : str
        Player position.
    current_week : int
        Current week in the season.

    Returns
    -------
    list[dict]
        Simulated trajectory from current_week to week 18.
    """
    # Distribute games evenly over remaining weeks
    remaining_weeks = 18 - current_week + 1
    games_per_week = target_games / remaining_weeks if remaining_weeks > 0 else 0

    # Build synthetic weekly stats
    weekly_stats = []
    for week in range(current_week, 19):
        # Roughly one game per week, adjust based on target_games
        if week <= current_week + target_games - 1:
            weekly_stats.append({
                "week": week,
                "fp": target_ppg,
                "games": 1,
            })
        else:
            weekly_stats.append({
                "week": week,
                "fp": 0,
                "games": 0,
            })

    return rollout_season(
        model=model,
        clip_bounds=clip_bounds,
        start_ktc=start_ktc,
        weekly_stats=weekly_stats,
        age=age,
        position=position,
        start_week=current_week,
        end_week=18,
    )
