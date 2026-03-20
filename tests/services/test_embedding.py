import importlib

import pytest
from httpx import ASGITransport, AsyncClient
from unittest.mock import MagicMock, patch


@pytest.fixture
def mock_model():
    """Mock SentenceTransformer to avoid loading real model in tests.

    We patch at the source (sentence_transformers.SentenceTransformer) so that
    the `from sentence_transformers import SentenceTransformer` statement inside
    `importlib.reload(app)` resolves to the mock even after the reload.
    """
    with patch("sentence_transformers.SentenceTransformer") as mock_cls:
        mock_instance = MagicMock()
        mock_cls.return_value = mock_instance
        yield mock_instance


@pytest.fixture
async def client(mock_model):
    import app
    importlib.reload(app)
    # Simulate startup: set the global model as the mock instance so that
    # /health returns "ok" (ASGITransport does not trigger ASGI lifespan events).
    app.model = mock_model
    transport = ASGITransport(app=app.app)
    async with AsyncClient(transport=transport, base_url="http://test") as c:
        yield c


async def test_health_returns_200(client):
    response = await client.get("/health")
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "ok"


async def test_health_returns_503_when_loading(mock_model):
    import app
    importlib.reload(app)
    # Leave app.model as None to simulate the loading/startup state.
    app.model = None
    transport = ASGITransport(app=app.app)
    async with AsyncClient(transport=transport, base_url="http://test") as c:
        response = await c.get("/health")
    assert response.status_code == 503
    data = response.json()
    assert data["status"] == "loading"
