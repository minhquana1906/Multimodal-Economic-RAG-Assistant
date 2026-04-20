import pytest
import pytest_httpx
from unittest.mock import AsyncMock, MagicMock, patch


@pytest.mark.asyncio
async def test_retriever_returns_empty_on_error():
    """Returns empty list when Qdrant is unreachable."""
    from orchestrator.services.retriever import RetrieverClient

    retriever = RetrieverClient("http://qdrant:6333", "test_collection")
    with patch.object(
        retriever.client,
        "query_points",
        new_callable=AsyncMock,
        side_effect=Exception("connection refused"),
    ):
        result = await retriever.hybrid_search(dense_vector=[0.1] * 1024)
    assert result == []


@pytest.mark.asyncio
async def test_retriever_hybrid_passes_sparse_vector():
    """When sparse_vector is provided, query_points is called with Prefetch + FusionQuery."""
    from orchestrator.services.retriever import RetrieverClient, SPARSE_VECTOR_NAME
    from qdrant_client.models import Fusion, FusionQuery

    retriever = RetrieverClient("http://qdrant:6333", "test_collection")

    mock_point = MagicMock()
    mock_point.id = "abc123"
    mock_point.score = 0.9
    mock_point.payload = {
        "text": "test",
        "source": "s",
        "title": "t",
        "url": "https://example.com/t",
    }

    mock_result = MagicMock()
    mock_result.points = [mock_point]

    with patch.object(
        retriever.client,
        "query_points",
        new_callable=AsyncMock,
        return_value=mock_result,
    ) as mock_qp:
        result = await retriever.hybrid_search(
            dense_vector=[0.1] * 1024,
            sparse_vector={"indices": [1, 42], "values": [0.22, 0.8]},
            top_k=5,
        )

    call_kwargs = mock_qp.call_args.kwargs
    assert "prefetch" in call_kwargs
    assert len(call_kwargs["prefetch"]) == 2
    assert call_kwargs["prefetch"][1].using == SPARSE_VECTOR_NAME
    assert isinstance(call_kwargs["query"], FusionQuery)
    assert call_kwargs["query"].fusion == Fusion.RRF

    assert len(result) == 1
    assert result[0]["id"] == "abc123"
    assert result[0]["score"] == 0.9
    assert result[0]["url"] == "https://example.com/t"


@pytest.mark.asyncio
async def test_retriever_dense_only_uses_named_vector():
    """Dense-only path (sparse_vector=None) uses DENSE_VECTOR_NAME."""
    from orchestrator.services.retriever import DENSE_VECTOR_NAME, RetrieverClient

    retriever = RetrieverClient("http://qdrant:6333", "test_collection")

    mock_point = MagicMock()
    mock_point.id = "xyz"
    mock_point.score = 0.8
    mock_point.payload = {
        "text": "content",
        "source": "src",
        "title": "ttl",
        "url": "https://example.com/ttl",
    }

    mock_result = MagicMock()
    mock_result.points = [mock_point]

    with patch.object(
        retriever.client,
        "query_points",
        new_callable=AsyncMock,
        return_value=mock_result,
    ) as mock_qp:
        result = await retriever.hybrid_search(dense_vector=[0.1] * 1024)

    call_kwargs = mock_qp.call_args.kwargs
    assert call_kwargs.get("using") == DENSE_VECTOR_NAME
    assert "prefetch" not in call_kwargs
    assert len(result) == 1
    assert result[0]["id"] == "xyz"
    assert result[0]["url"] == "https://example.com/ttl"


# InferenceClient tests
# Only mock requests to the inference host; let Qdrant version-check traffic pass through.
_only_inference = pytest.mark.httpx_mock(
    should_mock=lambda request: request.url.host == "inference"
)


@_only_inference
@pytest.mark.asyncio
async def test_inference_client_fetches_dense_and_sparse(httpx_mock):
    from orchestrator.services.inference import InferenceClient

    httpx_mock.add_response(json={"embeddings": [[0.1, 0.2]]})
    httpx_mock.add_response(json={"vectors": [{"indices": [1], "values": [0.9]}]})

    client = InferenceClient("http://inference:8001", timeout=30.0)
    dense = await client.embed_query("GDP Việt Nam?")
    sparse = await client.sparse_query("GDP Việt Nam?")

    assert dense == [0.1, 0.2]
    assert sparse == {"indices": [1], "values": [0.9]}


@_only_inference
@pytest.mark.asyncio
async def test_inference_client_embed_documents(httpx_mock):
    """embed_documents returns all embeddings for a batch of texts."""
    from orchestrator.services.inference import InferenceClient

    httpx_mock.add_response(json={"embeddings": [[0.1, 0.2], [0.3, 0.4]]})

    client = InferenceClient("http://inference:8001", timeout=30.0)
    result = await client.embed_documents(["text one", "text two"])

    assert result == [[0.1, 0.2], [0.3, 0.4]]


@_only_inference
@pytest.mark.asyncio
async def test_inference_client_sparse_documents(httpx_mock):
    """sparse_documents returns all sparse vectors for a batch of texts."""
    from orchestrator.services.inference import InferenceClient

    httpx_mock.add_response(
        json={"vectors": [{"indices": [1, 2], "values": [0.5, 0.8]}, {"indices": [3], "values": [0.3]}]}
    )

    client = InferenceClient("http://inference:8001", timeout=30.0)
    result = await client.sparse_documents(["text one", "text two"])

    assert result == [{"indices": [1, 2], "values": [0.5, 0.8]}, {"indices": [3], "values": [0.3]}]


@_only_inference
@pytest.mark.asyncio
async def test_inference_client_rerank(httpx_mock):
    """rerank returns a list of float scores."""
    from orchestrator.services.inference import InferenceClient

    httpx_mock.add_response(json={"scores": [0.9, 0.4, 0.7]})

    client = InferenceClient("http://inference:8001", timeout=30.0)
    scores = await client.rerank("query text", ["passage A", "passage B", "passage C"])

    assert scores == [0.9, 0.4, 0.7]
