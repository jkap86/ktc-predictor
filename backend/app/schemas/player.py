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


class PredictionResponse(BaseModel):
    player_id: str
    name: str
    position: str
    current_ktc: float
    predicted_ktc: float
    ktc_change: float
    ktc_change_pct: float
    confidence: Optional[float] = None


class CompareRequest(BaseModel):
    player_ids: list[str]


class PlayerComparison(BaseModel):
    player_id: str
    name: str
    position: str
    current_ktc: float
    predicted_ktc: float
    ktc_change: float
    seasons: list[PlayerSeason] = []


class CompareResponse(BaseModel):
    players: list[PlayerComparison]


class PredictionWithPPG(BaseModel):
    player_id: str
    name: str
    position: str
    ppg: float
    predicted_ktc: float
    current_ktc: float
    ktc_change_pct: float


# ========== Weekly Simulation Schemas ==========


class SimulateCurveRequest(BaseModel):
    games: int


class CurvePoint(BaseModel):
    ppg: int
    predicted_ktc: float


class SimulateCurveResponse(BaseModel):
    player_id: str
    name: str
    position: str
    starting_ktc: float
    current_ppg: float
    games: int
    curve: list[CurvePoint]


class WeeklyProjection(BaseModel):
    week: int
    ktc: float
    fp: float
    change: float


class SimulationResponse(BaseModel):
    player_id: str
    name: str
    position: str
    starting_ktc: float
    final_ktc: float
    total_change: float
    total_change_pct: float
    games: int
    ppg: float
    trajectory: list[WeeklyProjection]
