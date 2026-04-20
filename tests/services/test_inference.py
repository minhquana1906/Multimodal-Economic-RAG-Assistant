import importlib
from unittest.mock import MagicMock, patch

import pytest
from httpx import ASGITransport, AsyncClient


@pytest.fixture
def mock_flag_embedding():
    with (
        patch("FlagEmbedding.BGEM3FlagModel") as mock_embedder_cls,
        patch("FlagEmbedding.FlagReranker") as mock_reranker_cls,
    ):
        mock_embedder = MagicMock()
        mock_embedder.model_name = "BAAI/bge-m3"

        def _encode(texts, **_kwargs):
            return {
                "dense_vecs": [[0.1, 0.2, 0.3] for _ in texts],
                "lexical_weights": [{5: 0.5, 1: 0.1} for _ in texts],
            }

        mock_embedder.encode.side_effect = _encode
        mock_embedder_cls.return_value = mock_embedder

        mock_reranker = MagicMock()
        mock_reranker.model_name = "BAAI/bge-reranker-v2-m3"
        mock_reranker.compute_score.return_value = [-10.0, 10.0]
        mock_reranker_cls.return_value = mock_reranker

        yield mock_embedder, mock_reranker


@pytest.fixture
async def client(mock_flag_embedding):
    import inference_app

    importlib.reload(inference_app)
    mock_embedder, mock_reranker = mock_flag_embedding
    inference_app.embedder = mock_embedder
    inference_app.reranker = mock_reranker

    transport = ASGITransport(app=inference_app.app)
    async with AsyncClient(transport=transport, base_url="http://test") as c:
        yield c


async def test_health_reports_bge_models(client):
    response = await client.get("/health")
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "ok"
    assert data["embedding_model"] == "BAAI/bge-m3"
    assert data["reranker_model"] == "BAAI/bge-reranker-v2-m3"


async def test_sparse_returns_indices_and_values(client):
    response = await client.post("/sparse", json={"texts": ["hello"]})
    assert response.status_code == 200
    data = response.json()
    assert len(data["vectors"]) == 1
    sparse = data["vectors"][0]
    assert sparse["indices"] == [1, 5]
    assert sparse["values"] == [0.1, 0.5]


async def test_rerank_returns_same_length_scores(client):
    response = await client.post(
        "/rerank",
        json={"query": "q", "passages": ["a", "b"]},
    )
    assert response.status_code == 200
    data = response.json()
    assert len(data["scores"]) == 2
    assert all(0.0 <= s <= 1.0 for s in data["scores"])
    assert data["scores"][0] < 0.5
    assert data["scores"][1] > 0.5
