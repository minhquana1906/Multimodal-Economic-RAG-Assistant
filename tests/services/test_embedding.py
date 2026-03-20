import pytest
from unittest.mock import patch, MagicMock
from httpx import AsyncClient, ASGITransport


@pytest.fixture
def mock_model():
    """Mock SentenceTransformer to avoid loading real model in tests.

    We patch at the source (sentence_transformers.SentenceTransformer) so that
    the `from sentence_transformers import SentenceTransformer` statement inside
    `importlib.reload(main)` resolves to the mock even after the reload.
    """
    with patch("sentence_transformers.SentenceTransformer") as mock_cls:
        mock_instance = MagicMock()
        mock_cls.return_value = mock_instance
        yield mock_instance


@pytest.fixture
async def client(mock_model):
    import importlib
    import main
    importlib.reload(main)
    # Simulate startup: set the global model as the mock instance so that
    # /health returns "ok" (ASGITransport does not trigger ASGI lifespan events).
    main.model = mock_model
    transport = ASGITransport(app=main.app)
    async with AsyncClient(transport=transport, base_url="http://test") as c:
        yield c


@pytest.mark.asyncio
async def test_health_returns_200(client):
    response = await client.get("/health")
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "ok"
