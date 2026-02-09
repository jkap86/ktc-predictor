from pydantic import BaseModel
from typing import Optional


class WeeklyStat(BaseModel):
    week: int
    fantasy_points: float
    games_played: int
    snap_pct: float


class WeeklyKTC(BaseModel):
    week: int
    ktc: float
    date: str


class PlayerSeason(BaseModel):
    year: int
    age: int
    years_exp: int
    start_ktc: float
    end_ktc: float
    fantasy_points: float
    games_played: int
    ktc_30d_trend: Optional[float] = None
    ktc_90d_trend: Optional[float] = None
    ktc_volatility: float
    prior_year_fp: Optional[float] = None
    prior_year_games: Optional[int] = None
    fp_change_yoy: Optional[float] = None
    start_position_rank: int
    weekly_stats: list[WeeklyStat] = []
    weekly_ktc: list[WeeklyKTC] = []


class Player(BaseModel):
    player_id: str
    name: str
    position: str
    seasons: list[PlayerSeason] = []


class PlayerSummary(BaseModel):
    player_id: str
    name: str
    position: str
    latest_ktc: Optional[float] = None


class PlayerList(BaseModel):
    players: list[PlayerSummary]
    total: int


class EOSPredictionResponse(BaseModel):
    model_config = {"protected_namespaces": ()}

    player_id: str | None = None
    name: str | None = None
    position: str
    start_ktc: float
    predicted_end_ktc: float
    predicted_delta_ktc: float
    predicted_pct_change: float
    low_end_ktc: float | None = None
    high_end_ktc: float | None = None
    model_version: str = "eos_hgb_v1"
    anchor_year: int | None = None
    anchor_source: str | None = None
    baseline_year: int | None = None
    # Weekly blend fields (only populated when blend_weekly=true)
    eos_end_ktc: float | None = None
    weekly_end_ktc: float | None = None
    blend_weight: float | None = None
    games_played: int | None = None


class EOSPredictRequest(BaseModel):
    position: str
    start_ktc: float
    games_played: int
    ppg: float
    age: float | None = None
    weeks_missed: float | None = None
    draft_pick: float | None = None
    years_remaining: float | None = None


class CompareRequest(BaseModel):
    player_ids: list[str]
    blend_weekly: bool = False


class PlayerComparison(BaseModel):
    model_config = {"protected_namespaces": ()}

    player_id: str
    name: str
    position: str
    start_ktc: float
    predicted_end_ktc: float
    predicted_delta_ktc: float
    predicted_pct_change: float
    low_end_ktc: float | None = None
    high_end_ktc: float | None = None
    model_version: str = "eos_hgb_v1"
    anchor_year: int | None = None
    anchor_source: str | None = None
    baseline_year: int | None = None
    seasons: list[PlayerSeason] = []


class CompareResponse(BaseModel):
    players: list[PlayerComparison]


# Transition model schemas
class TrajectoryPoint(BaseModel):
    week: int
    ktc: float


class TrajectoryResponse(BaseModel):
    model_config = {"protected_namespaces": ()}

    player_id: str
    name: str
    position: str
    year: int
    start_ktc: float
    end_ktc: float
    delta_ktc: float
    delta_pct: float
    trajectory: list[TrajectoryPoint]
    actual_weekly_ktc: list[TrajectoryPoint] = []
    model_version: str = "transition_v1"


class WhatIfRequest(BaseModel):
    ppg: float
    games: int
    current_week: int = 1


class WhatIfResponse(BaseModel):
    model_config = {"protected_namespaces": ()}

    player_id: str
    name: str
    position: str
    start_ktc: float
    end_ktc: float
    delta_ktc: float
    delta_pct: float
    trajectory: list[TrajectoryPoint]
    scenario: WhatIfRequest
    model_version: str = "transition_v1"


class NextWeekRequest(BaseModel):
    position: str
    ktc_current: float
    ppg_cumulative: float
    games_played: int
    week: int
    weekly_fp: float = 0.0
    games_this_week: int = 0
    age: float | None = None


class NextWeekResponse(BaseModel):
    model_config = {"protected_namespaces": ()}

    position: str
    week: int
    ktc_next: float
    delta_log: float
    delta_pct: float
    model_version: str = "transition_v1"
