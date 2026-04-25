"""Test configuration — mock external dependencies."""

import os
from unittest.mock import AsyncMock, patch

import pytest

# Remove DATABASE_URL before any imports to prevent asyncpg pool creation
os.environ.pop("DATABASE_URL", None)


@pytest.fixture(autouse=True)
def mock_db():
    """Mock all DB calls so tests don't need a running postgres."""
    with patch("app.services.ktc_db.get_pool", new_callable=AsyncMock, side_effect=RuntimeError("DATABASE_URL not set")):
        with patch("app.services.ktc_db.get_latest_ktc", new_callable=AsyncMock, return_value=None):
            with patch("app.services.ktc_db.get_latest_ktc_batch", new_callable=AsyncMock, return_value={}):
                yield
