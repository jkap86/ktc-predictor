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
    """Compute prior-year KTC features for trajectory signal.

    Returns (prior_end_ktc, max_ktc_prior).
    """
    sorted_seasons = sorted(seasons, key=lambda s: s["year"])
    prior_end_ktc = None
    max_ktc_prior = None

    for season in sorted_seasons:
        year = season["year"]
        if year >= anchor_year:
            break
        if (season.get("years_exp") or 0) < 0:
            continue
        end_ktc = season.get("end_ktc")
        if _is_valid_ktc(end_ktc):
            prior_end_ktc = end_ktc
            if max_ktc_prior is None or end_ktc > max_ktc_prior:
                max_ktc_prior = end_ktc

    return prior_end_ktc, max_ktc_prior


def compute_prior_ppg(
    seasons: list[dict],
    anchor_year: int,
    min_games: int = 4,
) -> float | None:
    """Compute prior-year PPG for a player.

    Returns prior season's PPG, or None if no valid prior data.
    """
    for season in sorted(seasons, key=lambda s: s["year"], reverse=True):
        if season["year"] >= anchor_year:
            continue
        games = season.get("games_played", 0) or 0
        fp = season.get("fantasy_points", 0) or 0
        if games >= min_games:
            return fp / games
    return None


def compute_prior_behavioral_features(
    seasons: list[dict],
    anchor_year: int,
    min_games: int = 4,
) -> dict | None:
    """Pull 5 behavioral aggregates from the most-recent completed season
    before ``anchor_year``.

    Used by v3_hgb_prior_signals and later iterations that condition on a
    player's prior-year consistency, upside, and role.

    Returns a dict with weekly_fp_cv / boom_rate / bust_rate / snap_pct /
    ktc_volatility, or None if no prior season has >= min_games games.
    """
    for season in sorted(seasons, key=lambda s: s["year"], reverse=True):
        if season["year"] >= anchor_year:
            continue
        games = season.get("games_played", 0) or 0
        if games < min_games:
            continue
        return {
            "prior_weekly_fp_cv": float(season.get("weekly_fp_cv") or 0.0),
            "prior_boom_rate": float(season.get("boom_rate") or 0.0),
            "prior_bust_rate": float(season.get("bust_rate") or 0.0),
            "prior_snap_pct": float(season.get("snap_pct") or 0.0),
            "prior_ktc_volatility": float(season.get("ktc_volatility") or 0.0),
        }
    return None


def compute_prior_position_stats(
    seasons: list[dict],
    position: str,
    anchor_year: int,
    min_games: int = 4,
) -> dict | None:
    """Pull position-specific stats from the most-recent completed season
    before ``anchor_year``. Used by v4+."""
    for season in sorted(seasons, key=lambda s: s["year"], reverse=True):
        if season["year"] >= anchor_year:
            continue
        games = season.get("games_played", 0) or 0
        if games < min_games:
            continue
        if position == "QB":
            return {
                "prior_passing_tds": float(season.get("passing_tds") or 0),
                "prior_interceptions": float(season.get("interceptions") or 0),
            }
        elif position == "RB":
            return {
                "prior_carries": float(season.get("carries") or 0),
                "prior_red_zone_touches": float(season.get("red_zone_touches") or 0),
            }
        elif position in ("WR", "TE"):
            return {
                "prior_targets": float(season.get("targets") or 0),
                "prior_red_zone_targets": float(season.get("red_zone_targets") or 0),
            }
    return None


def compute_momentum_features(
    seasons: list[dict],
    anchor_year: int,
) -> dict | None:
    """Pull KTC momentum features from the anchor year's season record."""
    for season in sorted(seasons, key=lambda s: s["year"], reverse=True):
        if season["year"] == anchor_year:
            return {
                "ktc_30d_trend": float(season.get("ktc_30d_trend") or 0),
                "ktc_90d_trend": float(season.get("ktc_90d_trend") or 0),
                "momentum_ratio": float(season.get("momentum_ratio") or 0),
                "max_games_missed_streak": float(season.get("max_games_missed_streak") or 0),
            }
    return None
