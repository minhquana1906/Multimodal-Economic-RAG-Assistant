import asyncio
import importlib
import io

import pytest
import torch
import torchaudio
from httpx import ASGITransport, AsyncClient
from unittest.mock import AsyncMock, MagicMock, patch


@pytest.fixture
def mock_qwen_asr_model():
    """Mock Qwen3ASRModel to avoid loading real model in tests."""
    mock_instance = MagicMock()
    mock_instance.transcribe = MagicMock()
    return mock_instance


@pytest.fixture
async def client(mock_qwen_asr_model):
    """ASGI test client with mocked model (model NOT loaded by default)."""
    with patch.dict("os.environ", {"ASR_MODEL": "Qwen/Qwen3-ASR-1.7B"}):
        import asr_app
        importlib.reload(asr_app)
        transport = ASGITransport(app=asr_app.app)
        async with AsyncClient(transport=transport, base_url="http://test") as c:
            yield c


async def test_health_returns_idle_when_model_not_loaded(client):
    """Health returns 200 with status='idle' when model is not loaded."""
    response = await client.get("/health")
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "idle"
    assert data["model_loaded"] is False


async def test_health_returns_ok_when_model_loaded(mock_qwen_asr_model):
    """Health returns 200 with status='ok' when model is loaded."""
    import asr_app
    importlib.reload(asr_app)
    asr_app.on_demand.model = mock_qwen_asr_model
    transport = ASGITransport(app=asr_app.app)
    async with AsyncClient(transport=transport, base_url="http://test") as c:
        response = await c.get("/health")
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "ok"
    assert data["model_loaded"] is True


async def test_health_returns_loading_during_model_load(mock_qwen_asr_model):
    """Health returns 200 with status='loading' when model is being loaded."""
    import asr_app
    importlib.reload(asr_app)
    asr_app.on_demand._loading = True
    transport = ASGITransport(app=asr_app.app)
    async with AsyncClient(transport=transport, base_url="http://test") as c:
        response = await c.get("/health")
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "loading"
    assert data["model_loaded"] is False
