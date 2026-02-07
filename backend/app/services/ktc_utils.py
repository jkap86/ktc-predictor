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
