"""Standalone FastAPI app for KTC prediction.

Usage:
    cd backend
    uvicorn api.main:app --port 8002
"""

import sys
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI, HTTPException, Request
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field, field_validator

# Ensure ktc_model is importable
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from ktc_model.io import load_bundle
from ktc_model.predict import predict_end_ktc, VALID_POSITIONS

_state: dict = {}


@asynccontextmanager
async def lifespan(app: FastAPI):
    model_dir = Path(__file__).resolve().parent.parent / "models"
    try:
        bundle = load_bundle(str(model_dir))
        _state["models"] = bundle["models"]
        _state["clip_bounds"] = bundle["clip_bounds"]
        _state["calibrators"] = bundle["calibrators"]
        _state["sentinel_impute"] = bundle.get("sentinel_impute", {})
    except Exception as e:
        print(f"WARNING: Failed to load models: {e}")
        _state["models"] = {}
        _state["clip_bounds"] = {}
        _state["calibrators"] = {}
        _state["sentinel_impute"] = {}
    yield
    _state.clear()


app = FastAPI(title="KTC Predictor", lifespan=lifespan)


@app.exception_handler(RequestValidationError)
async def validation_exception_handler(request: Request, exc: RequestValidationError):
    errors = exc.errors()
    messages = []
    for err in errors:
        loc = " -> ".join(str(l) for l in err.get("loc", []))
        messages.append(f"{loc}: {err.get('msg', '')}")
    return JSONResponse(status_code=400, content={"detail": "; ".join(messages)})


class PredictRequest(BaseModel):
    position: str
    games_played: int = Field(..., alias="gamesPlayed", ge=0)
    ppg: float = Field(..., ge=0)
    start_ktc: float = Field(..., alias="startKtc", gt=0)
    age: int | None = None
    weeks_missed: int | None = Field(None, alias="weeksMissed", ge=0)
    draft_pick: int | None = Field(None, alias="draftPick", ge=1)
    years_remaining: int | None = Field(None, alias="yearsRemaining", ge=0)

    model_config = {"populate_by_name": True}

    @field_validator("position")
    @classmethod
    def validate_position(cls, v: str) -> str:
        v = v.upper()
        if v not in VALID_POSITIONS:
            raise ValueError(
                f"Invalid position '{v}'. Must be one of {sorted(VALID_POSITIONS)}"
            )
        return v


class PredictResponse(BaseModel):
    position: str
    games_played: int = Field(..., alias="gamesPlayed")
    ppg: float
    start_ktc: float = Field(..., alias="startKtc")
    age: int | None = None
    weeks_missed: int | None = Field(None, alias="weeksMissed")
    draft_pick: int | None = Field(None, alias="draftPick")
    years_remaining: int | None = Field(None, alias="yearsRemaining")
    predicted_delta_ktc: float = Field(..., alias="predictedDeltaKtc")
    predicted_end_ktc: float = Field(..., alias="predictedEndKtc")

    model_config = {"populate_by_name": True, "serialize_by_alias": True}


@app.get("/health")
async def health():
    positions = sorted(_state.get("models", {}).keys())
    if not positions:
        raise HTTPException(status_code=503, detail="No models loaded")
    return {"status": "healthy", "positions": positions}


@app.post("/ktc/predict", response_model=PredictResponse)
async def predict(req: PredictRequest):
    try:
        result = predict_end_ktc(
            models=_state["models"],
            clip_bounds=_state["clip_bounds"],
            calibrators=_state["calibrators"],
            position=req.position,
            gp=req.games_played,
            ppg=req.ppg,
            start_ktc=req.start_ktc,
            age=req.age,
            weeks_missed=req.weeks_missed,
            draft_pick=req.draft_pick,
            years_remaining=req.years_remaining,
            sentinel_impute=_state.get("sentinel_impute"),
        )
    except (ValueError, KeyError) as e:
        raise HTTPException(status_code=400, detail=str(e))

    return PredictResponse(
        position=req.position,
        games_played=req.games_played,
        ppg=req.ppg,
        start_ktc=req.start_ktc,
        age=req.age,
        weeks_missed=req.weeks_missed,
        draft_pick=req.draft_pick,
        years_remaining=req.years_remaining,
        predicted_delta_ktc=result["delta_ktc"],
        predicted_end_ktc=result["end_ktc"],
    )
