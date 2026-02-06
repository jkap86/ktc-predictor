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
    latest_year: Optional[int] = None


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
    model_version: str = "eos_hgb_v1"


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


class PlayerComparison(BaseModel):
    model_config = {"protected_namespaces": ()}

    player_id: str
    name: str
    position: str
    start_ktc: float
    predicted_end_ktc: float
    predicted_delta_ktc: float
    predicted_pct_change: float
    model_version: str = "eos_hgb_v1"
    seasons: list[PlayerSeason] = []


class CompareResponse(BaseModel):
    players: list[PlayerComparison]
