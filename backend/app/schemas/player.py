from pydantic import BaseModel, Field
from typing import Literal, Optional


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
    age: Optional[int] = None
    years_exp: int = 0
    start_ktc: float = 0
    end_ktc: float = 0
    fantasy_points: float = 0
    games_played: int = 0
    ktc_30d_trend: Optional[float] = None
    ktc_90d_trend: Optional[float] = None
    ktc_volatility: float = 0
    prior_year_fp: Optional[float] = None
    prior_year_games: Optional[int] = None
    fp_change_yoy: Optional[float] = None
    start_position_rank: Optional[int] = None
    weekly_stats: list[WeeklyStat] = []
    weekly_ktc: list[WeeklyKTC] = []


class Player(BaseModel):
    player_id: str
    name: str
    position: str
    seasons: list[PlayerSeason] = []
    live_ktc: float | None = None


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
    model_version: str = ""
    anchor_year: int | None = None
    anchor_source: str | None = None
    baseline_year: int | None = None


class EOSPredictRequest(BaseModel):
    position: Literal["QB", "RB", "WR", "TE"]
    start_ktc: float = Field(ge=0, le=99999)
    games_played: int = Field(ge=0, le=17)
    ppg: float = Field(ge=0)
    age: float | None = Field(None, ge=18, le=50)
    weeks_missed: float | None = Field(None, ge=0, le=17)
    draft_pick: float | None = Field(None, ge=1, le=260)
    years_remaining: float | None = Field(None, ge=0, le=6)


class ModelInfo(BaseModel):
    id: str
    name: str
    description: str = ""
    created: str = ""
    # int for iterations with a single feature count across positions,
    # dict[str, int] for iterations where positions use different feature sets
    # (e.g., v3 uses 26 features for RB/WR, 20 for QB/TE).
    features_count: int | dict[str, int] | None = None
    positions: list[str] = []
    is_default: bool = False
    metrics: dict = {}


class ModelsListResponse(BaseModel):
    models: list[ModelInfo]
    default_model: str


class EOSBatchRequest(BaseModel):
    position: Literal["QB", "RB", "WR", "TE"]
    start_ktc: float = Field(ge=0, le=99999)
    games_played: int = Field(ge=0, le=17)
    ppg_values: list[float] = Field(description="List of PPG values to predict for")
    age: float | None = Field(None, ge=18, le=50)
    weeks_missed: float | None = Field(None, ge=0, le=17)
    draft_pick: float | None = Field(None, ge=1, le=260)
    years_remaining: float | None = Field(None, ge=0, le=6)


class EOSBatchResponse(BaseModel):
    model_config = {"protected_namespaces": ()}

    predictions: list[EOSPredictionResponse]
    model_version: str = ""


class PlayerWhatIfBatchRequest(BaseModel):
    games_played: int = Field(ge=0, le=17)
    ppg_values: list[float] = Field(description="List of PPG values to predict for")


class HistoricalPrediction(BaseModel):
    model_config = {"protected_namespaces": ()}

    year: int
    model_version: str
    start_ktc: float
    actual_end_ktc: float | None = None
    predicted_end_ktc: float
    predicted_delta_ktc: float
    predicted_pct_change: float
    low_end_ktc: float | None = None
    high_end_ktc: float | None = None
    error: float | None = None
    games_played: int
    ppg: float


class HistoricalResponse(BaseModel):
    player_id: str
    name: str
    position: str
    predictions: list[HistoricalPrediction]
