import importlib

import numpy as np
import pytest
from httpx import ASGITransport, AsyncClient
from unittest.mock import MagicMock, patch


@pytest.fixture
def mock_model():
    """Mock SentenceTransformer to avoid loading real model in tests.

    We patch at the source (sentence_transformers.SentenceTransformer) so that
    the `from sentence_transformers import SentenceTransformer` statement inside
    `importlib.reload(embedding_app)` resolves to the mock even after the reload.
    """
    with patch("sentence_transformers.SentenceTransformer") as mock_cls:
        mock_instance = MagicMock()
        mock_cls.return_value = mock_instance
        yield mock_instance


@pytest.fixture
async def client(mock_model):
    import embedding_app
    importlib.reload(embedding_app)
    # Simulate startup: set the global model as the mock instance so that
    # /health returns "ok" (ASGITransport does not trigger ASGI lifespan events).
    embedding_app.model = mock_model
    transport = ASGITransport(app=embedding_app.app)
    async with AsyncClient(transport=transport, base_url="http://test") as c:
        yield c


async def test_health_returns_200(client):
    response = await client.get("/health")
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "ok"


async def test_health_returns_503_when_loading(mock_model):
    import embedding_app
    importlib.reload(embedding_app)
    # Leave embedding_app.model as None to simulate the loading/startup state.
    embedding_app.model = None
    transport = ASGITransport(app=embedding_app.app)
    async with AsyncClient(transport=transport, base_url="http://test") as c:
        response = await c.get("/health")
    assert response.status_code == 503
    data = response.json()
    assert data["status"] == "loading"


async def test_embed_documents(client, mock_model):
    mock_model.encode.return_value = np.array([[0.1] * 1024, [0.2] * 1024])
    response = await client.post("/embed", json={
        "texts": ["Kinh tế Việt Nam", "Lạm phát tăng"],
        "is_query": False,
    })
    assert response.status_code == 200
    data = response.json()
    assert len(data["embeddings"]) == 2
    assert len(data["embeddings"][0]) == 1024
    mock_model.encode.assert_called_once()
    # Document encoding: no prompt_name
    call_kwargs = mock_model.encode.call_args
    assert "prompt_name" not in call_kwargs.kwargs or call_kwargs.kwargs.get("prompt_name") is None


async def test_embed_queries(client, mock_model):
    mock_model.encode.return_value = np.array([[0.5] * 1024])
    response = await client.post("/embed", json={
        "texts": ["GDP Việt Nam 2024"],
        "is_query": True,
    })
    assert response.status_code == 200
    data = response.json()
    assert len(data["embeddings"]) == 1
    assert len(data["embeddings"][0]) == 1024
    call_kwargs = mock_model.encode.call_args
    assert call_kwargs.kwargs.get("prompt_name") == "query"


async def test_embed_empty_texts(client, mock_model):
    response = await client.post("/embed", json={
        "texts": [],
        "is_query": False,
    })
    assert response.status_code == 422
