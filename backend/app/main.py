import logging
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse

from dotenv import load_dotenv
load_dotenv()

from app.config import CORS_ORIGINS, DEFAULT_MODEL_ID
from app.routers import players_router, predictions_router, models_router
from app.services.model_registry import get_registry
from app.services.ktc_db import get_pool, close_pool
from app.services.prediction_cache import get_prediction_cache

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

    # Precompute predictions for all players (used by search endpoint)
    try:
        get_prediction_cache(default)
    except Exception:
        logger.exception("Failed to build prediction cache")

    # Initialize DB pool for live KTC lookups
    try:
        await get_pool()
        logger.info("Database pool initialized")
    except Exception:
        logger.warning("Database pool not available — will use training-data KTC values")

    yield

    await close_pool()
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


# ── Static frontend serving ──────────────────────────────────────────────
# Serve the Next.js static export from frontend/out/ if it exists.
# This must come AFTER API routes so /api/* takes precedence.

FRONTEND_DIR = Path(__file__).resolve().parent.parent.parent / "frontend" / "out"

if FRONTEND_DIR.is_dir():
    # Serve static assets (_next/, images, etc.)
    app.mount("/_next", StaticFiles(directory=FRONTEND_DIR / "_next"), name="next_static")

    # SPA catch-all: serve index.html for any non-API, non-static route
    @app.get("/{full_path:path}")
    async def serve_frontend(request: Request, full_path: str):
        # Try exact file first (e.g. /favicon.ico, /robots.txt)
        file_path = FRONTEND_DIR / full_path
        if file_path.is_file():
            return FileResponse(file_path)

        # Try directory with index.html (e.g. /player/123/ -> /player/[id]/index.html)
        index_path = file_path / "index.html"
        if index_path.is_file():
            return FileResponse(index_path)

        # Dynamic routes: /player/anything/ -> /player/[id]/index.html
        parts = full_path.strip("/").split("/")
        if len(parts) >= 2 and parts[0] == "player":
            dynamic_index = FRONTEND_DIR / "player" / "[id]" / "index.html"
            if dynamic_index.is_file():
                return FileResponse(dynamic_index)

        # Fallback to root index.html (SPA client-side routing)
        root_index = FRONTEND_DIR / "index.html"
        if root_index.is_file():
            return FileResponse(root_index)

        return {"message": "KTC Predictor Dev API", "docs": "/docs"}
else:
    @app.get("/")
    def root():
        return {"message": "KTC Predictor Dev API", "docs": "/docs"}
