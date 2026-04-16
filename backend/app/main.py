import logging
from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from app.config import CORS_ORIGINS, DEFAULT_MODEL_ID
from app.routers import players_router, predictions_router, models_router
from app.services.model_registry import get_registry

logger = logging.getLogger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI):
    logger.info("Initializing model registry...")
    registry = get_registry()
    ids = registry.list_ids()
    logger.info("Discovered %d model iteration(s): %s", len(ids), ids)

    if DEFAULT_MODEL_ID in ids:
        registry.set_default(DEFAULT_MODEL_ID)
    elif ids:
        logger.warning(
            "Default model '%s' not found, using '%s'",
            DEFAULT_MODEL_ID,
            ids[0],
        )

    # Eagerly load the default model bundle
    try:
        default = registry.get()
        default.load_bundle()
        metrics = default.get_metrics()
        logger.info("Default model '%s' loaded. Metrics: %s", default.id, metrics)
    except Exception:
        logger.exception("Failed to load default model")
        raise

    yield
    logger.info("Shutting down...")


app = FastAPI(
    title="KTC Predictor Dev API",
    description="Fantasy Football KTC Value Prediction API (Multi-Model)",
    version="0.1.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=CORS_ORIGINS,
    allow_credentials=True,
    allow_methods=["GET", "POST", "OPTIONS"],
    allow_headers=["Content-Type"],
)

app.include_router(players_router)
app.include_router(predictions_router)
app.include_router(models_router)


@app.get("/")
def root():
    return {"message": "KTC Predictor Dev API", "docs": "/docs"}


@app.get("/health")
def health_check():
    try:
        registry = get_registry()
        default = registry.get()
        ready = default._bundle is not None
    except Exception:
        ready = False
    return {
        "status": "healthy" if ready else "degraded",
        "model_ready": ready,
    }
