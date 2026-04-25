"""API smoke tests + prediction consistency tests."""

import pytest
import httpx


@pytest.fixture(scope="module")
def anyio_backend():
    return "asyncio"


@pytest.fixture(scope="module")
async def client():
    from app.main import app
    transport = httpx.ASGITransport(app=app)
    async with httpx.AsyncClient(transport=transport, base_url="http://test") as c:
        yield c


# ── Smoke tests ──────────────────────────────────────────────────────────

@pytest.mark.anyio
async def test_health(client):
    r = await client.get("/health")
    assert r.status_code == 200
    assert "status" in r.json()


@pytest.mark.anyio
async def test_models_list(client):
    r = await client.get("/api/models")
    assert r.status_code == 200
    data = r.json()
    assert "models" in data
    assert "default_model" in data
    assert len(data["models"]) > 0


@pytest.mark.anyio
async def test_players_search(client):
    r = await client.get("/api/players?limit=10")
    assert r.status_code == 200
    data = r.json()
    assert "players" in data
    assert "total" in data
    assert len(data["players"]) <= 10
    assert data["total"] >= len(data["players"])


@pytest.mark.anyio
async def test_players_total_before_limit(client):
    """total should reflect all matches, not just the limited results."""
    r_all = await client.get("/api/players?limit=2000&sort_by=ktc&sort_order=desc")
    r_limited = await client.get("/api/players?limit=10&sort_by=ktc&sort_order=desc")
    all_total = r_all.json()["total"]
    limited_total = r_limited.json()["total"]
    assert limited_total == all_total


@pytest.mark.anyio
async def test_positions(client):
    r = await client.get("/api/players/positions")
    assert r.status_code == 200
    assert "positions" in r.json()


@pytest.mark.anyio
async def test_model_diagnostics(client):
    r = await client.get("/api/models/diagnostics")
    assert r.status_code == 200
    data = r.json()
    assert "model_id" in data
    assert "metrics" in data


# ── Invalid model returns 400 ───────────────────────────────────────────

@pytest.mark.anyio
async def test_invalid_model_players(client):
    r = await client.get("/api/players?model=nonexistent_model_xyz")
    assert r.status_code == 400


@pytest.mark.anyio
async def test_invalid_model_top_movers(client):
    r = await client.get("/api/top-movers?model=nonexistent_model_xyz")
    assert r.status_code == 400


# ── Prediction metadata ─────────────────────────────────────────────────

@pytest.mark.anyio
async def test_players_include_prediction_meta(client):
    """Table rows should include prediction metadata."""
    r = await client.get("/api/players?limit=5&sort_by=ktc&sort_order=desc")
    assert r.status_code == 200
    players = r.json()["players"]
    # Find a player with a prediction
    with_pred = [p for p in players if p.get("predicted_end_ktc") is not None]
    if with_pred:
        meta = with_pred[0].get("prediction_meta")
        assert meta is not None, "prediction_meta should be present"
        assert "model_id" in meta
        assert "start_ktc_used" in meta
        assert "ppg_used" in meta
        assert "ppg_source" in meta
        assert "ktc_source" in meta
