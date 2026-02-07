from contextlib import asynccontextmanager
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from app.config import CORS_ORIGINS
from app.routers import players_router, predictions_router
from app.services.model_service import get_model_service
from ktc_model.age_adjustment import env_flag


@asynccontextmanager
async def lifespan(app: FastAPI):
    # Startup: Initialize model
    print("Initializing KTC prediction model...")
    model_service = get_model_service()
    metrics = model_service.initialize()
    print(f"Model initialized. Metrics: {metrics}")
    if env_flag("KTC_ENABLE_AGE_DECLINE_ADJ"):
        print("Age decline adjustment ENABLED")
    yield
    # Shutdown: cleanup if needed
    print("Shutting down...")


app = FastAPI(
    title="KTC Predictor API",
    description="Fantasy Football KTC Value Prediction API",
    version="1.0.0",
    lifespan=lifespan,
)

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=CORS_ORIGINS,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Include routers
app.include_router(players_router)
app.include_router(predictions_router)


@app.get("/")
def root():
    return {"message": "KTC Predictor API", "docs": "/docs"}


@app.get("/health")
def health_check():
    return {"status": "healthy"}
