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


@pytest.mark.anyio
async def test_invalid_model_diagnostics(client):
    r = await client.get("/api/models/diagnostics?model=nonexistent_model_xyz")
    assert r.status_code == 400


# ── Prediction metadata ─────────────────────────────────────────────────

@pytest.mark.anyio
async def test_players_include_prediction_meta(client):
    """Table rows should include prediction metadata."""
    r = await client.get("/api/players?limit=5&sort_by=ktc&sort_order=desc")
    assert r.status_code == 200
    players = r.json()["players"]
    with_pred = [p for p in players if p.get("predicted_end_ktc") is not None]
    assert with_pred, "Expected at least one player with prediction data"
    meta = with_pred[0].get("prediction_meta")
    assert meta is not None, "prediction_meta should be present"
    assert "model_id" in meta
    assert "start_ktc_used" in meta
    assert "ppg_used" in meta
    assert "ppg_source" in meta
    assert "ktc_source" in meta


@pytest.mark.anyio
async def test_players_include_delta_and_pct(client):
    """Table rows should include predicted_delta_ktc and predicted_pct_change."""
    r = await client.get("/api/players?limit=5&sort_by=ktc&sort_order=desc")
    players = r.json()["players"]
    with_pred = [p for p in players if p.get("predicted_end_ktc") is not None]
    assert with_pred
    p = with_pred[0]
    assert "predicted_delta_ktc" in p
    assert "predicted_pct_change" in p


# ── Prediction consistency ──────────────────────────────────────────────

@pytest.mark.anyio
async def test_table_detail_prediction_consistency(client):
    """Table prediction and detail prediction should match for the same player/model.

    Both paths use build_prediction_inputs() with the same live KTC (mocked to None),
    same Sleeper projections (mocked to {}), and same model.
    """
    # Get default model ID
    models_r = await client.get("/api/models")
    default_model = models_r.json()["default_model"]

    # Get table predictions
    table_r = await client.get(f"/api/players?limit=20&sort_by=ktc&sort_order=desc&model={default_model}")
    assert table_r.status_code == 200
    players = table_r.json()["players"]

    with_pred = [p for p in players if p.get("predicted_end_ktc") is not None]
    assert with_pred, "Need at least one player with table prediction"

    # Pick first player with prediction
    table_player = with_pred[0]
    pid = table_player["player_id"]

    # Get detail prediction for the same player and model
    detail_r = await client.get(f"/api/players/{pid}/predict?model={default_model}")
    assert detail_r.status_code == 200
    detail = detail_r.json()

    # Core consistency: same model + same inputs = same prediction
    assert table_player["prediction_meta"]["model_id"] == detail.get("model_version") or detail.get("prediction_meta", {}).get("model_id")
    assert table_player["predicted_end_ktc"] == detail["predicted_end_ktc"], (
        f"Table predicted {table_player['predicted_end_ktc']} but detail predicted {detail['predicted_end_ktc']} "
        f"for player {pid}. Table meta: {table_player.get('prediction_meta')}, "
        f"Detail meta: {detail.get('prediction_meta')}"
    )
