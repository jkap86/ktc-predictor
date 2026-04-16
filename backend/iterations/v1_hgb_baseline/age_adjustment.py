"""Post-calibration age decline adjustment for KTC predictions."""

from __future__ import annotations

import os
from typing import Optional


def env_flag(name: str, default: bool = False) -> bool:
    """Check if an environment variable flag is enabled."""
    v = os.getenv(name)
    if v is None:
        return default
    return v.strip().lower() in {"1", "true", "yes", "y", "on"}


def apply_age_decline_adjustment(
    log_ratio: float,
    age: Optional[float],
    position: Optional[str],
) -> float:
    """
    Post-calibration adjustment in log-ratio space to reflect age decline.

    Applied AFTER calibration and BEFORE KTC-aware clamp.

    Rules:
      - RB: if age > 26 -> subtract 0.0065 * (age - 26)
      - QB: if age > 29 -> subtract 0.0045 * (age - 29)
      - WR: if age > 29 -> subtract 0.0015 * (age - 29)
      - TE: no adjustment

    Safe for None age/position.
    """
    if age is None or position is None:
        return log_ratio

    pos = position.upper().strip()

    # RB: strongest penalty
    if pos == "RB" and age > 26:
        return log_ratio - 0.0065 * (age - 26)

    # QB: moderate penalty
    if pos == "QB" and age > 29:
        return log_ratio - 0.0045 * (age - 29)

    # WR: subtle penalty
    if pos == "WR" and age > 29:
        return log_ratio - 0.0015 * (age - 29)

    return log_ratio
