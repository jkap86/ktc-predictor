"""Test configuration — mock external dependencies."""

import os
from unittest.mock import AsyncMock, patch

import pytest

# Remove DATABASE_URL before any imports to prevent asyncpg pool creation
os.environ.pop("DATABASE_URL", None)


@pytest.fixture(autouse=True)
def mock_external():
    """Mock DB and Sleeper API so tests are fast and deterministic.

    Patches at both module-level AND route-local import points to ensure
    all code paths use the mocks.
    """
    patches = [
        # Module-level DB functions
        patch("app.services.ktc_db.get_pool", new_callable=AsyncMock, side_effect=RuntimeError("no DB")),
        patch("app.services.ktc_db.get_latest_ktc", new_callable=AsyncMock, return_value=None),
        patch("app.services.ktc_db.get_latest_ktc_batch", new_callable=AsyncMock, return_value={}),
        # Route-local DB imports
        patch("app.routers.players.get_latest_ktc", new_callable=AsyncMock, return_value=None),
        patch("app.routers.players.get_latest_ktc_batch", new_callable=AsyncMock, return_value={}),
        patch("app.services.eos_model_service.get_latest_ktc", new_callable=AsyncMock, return_value=None),
        # Sleeper projections (no network calls)
        patch("app.services.sleeper.fetch_projections", return_value={}),
    ]
    # Clear Sleeper LRU cache so tests get the mocked version
    from app.services.sleeper import get_projections
    get_projections.cache_clear()

    started = [p.start() for p in patches]
    yield
    for p in patches:
        p.stop()
    get_projections.cache_clear()
