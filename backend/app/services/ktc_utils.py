"""Shared KTC selection utilities used by eos_model_service and data_loader."""


def _is_valid_ktc(x) -> bool:
    """Return True if x is a usable KTC value (not None, not zero, not a 9999 sentinel)."""
    return x is not None and 0 < x < 9999


def select_anchor_ktc(
    seasons: list[dict],
) -> tuple[float, int, str] | None:
    """Pick the best anchor KTC from a player's seasons.

    Iterates newest-first, preferring ``end_ktc`` then ``start_ktc``.

    Returns ``(value, year, source)`` where *source* is ``"end_ktc"``
    or ``"start_ktc"``, or ``None`` if nothing valid is found.
    """
    for season in sorted(seasons, key=lambda s: s["year"], reverse=True):
        # Skip pre-draft seasons (college data)
        if (season.get("years_exp") or 0) < 0:
            continue
        end = season.get("end_ktc")
        if _is_valid_ktc(end):
            return (end, season["year"], "end_ktc")
        start = season.get("start_ktc")
        if _is_valid_ktc(start):
            return (start, season["year"], "start_ktc")
    return None


def select_baseline_stats(
    seasons: list[dict],
) -> tuple[int, int, float] | None:
    """Find the most recent season with games played > 0.

    Returns ``(year, games, ppg)`` or ``None``.
    """
    played = [s for s in seasons if (s.get("games_played") or 0) > 0]
    if not played:
        return None
    baseline = max(played, key=lambda s: s["year"])
    games = baseline.get("games_played", 0) or 0
    fp = baseline.get("fantasy_points", 0) or 0.0
    ppg = fp / games if games > 0 else 0.0
    return (baseline["year"], games, ppg)


def compute_prior_ktc_features(
    seasons: list[dict],
    anchor_year: int,
) -> tuple[float | None, float | None]:
    """Compute prior-year KTC features for QB trajectory signal.

    For a given anchor year, looks at prior seasons to compute:
    - prior_end_ktc: The end_ktc from the season immediately before anchor_year
    - max_ktc_prior: The maximum end_ktc from all seasons before anchor_year

    Parameters
    ----------
    seasons : list[dict]
        List of player season dicts with 'year' and 'end_ktc' fields.
    anchor_year : int
        The year we're predicting for (current/latest season).

    Returns
    -------
    tuple[float | None, float | None]
        (prior_end_ktc, max_ktc_prior) - both None if no valid prior data.
    """
    # Sort seasons by year ascending
    sorted_seasons = sorted(seasons, key=lambda s: s["year"])

    prior_end_ktc = None
    max_ktc_prior = None

    for season in sorted_seasons:
        year = season["year"]
        if year >= anchor_year:
            # Stop at anchor year - we only want prior seasons
            break

        # Skip pre-draft seasons (college data)
        if (season.get("years_exp") or 0) < 0:
            continue

        end_ktc = season.get("end_ktc")
        if _is_valid_ktc(end_ktc):
            # Track the most recent prior season's end_ktc
            prior_end_ktc = end_ktc
            # Track career max
            if max_ktc_prior is None or end_ktc > max_ktc_prior:
                max_ktc_prior = end_ktc

    return prior_end_ktc, max_ktc_prior


def compute_prior_ppg(
    seasons: list[dict],
    anchor_year: int,
    min_games: int = 4,
) -> float | None:
    """Compute prior-year PPG for a player.

    Looks at the most recent season before anchor_year with sufficient games
    to compute a meaningful PPG.

    Parameters
    ----------
    seasons : list[dict]
        List of player season dicts with 'year', 'games_played', 'fantasy_points'.
    anchor_year : int
        The year we're predicting for (current/latest season).
    min_games : int
        Minimum games required for valid PPG (default: 4).

    Returns
    -------
    float or None
        Prior season's PPG, or None if no valid prior data.
    """
    # Sort by year descending to get most recent first
    for season in sorted(seasons, key=lambda s: s["year"], reverse=True):
        if season["year"] >= anchor_year:
            continue

        games = season.get("games_played", 0) or 0
        fp = season.get("fantasy_points", 0) or 0

        if games >= min_games:
            return fp / games

    return None
