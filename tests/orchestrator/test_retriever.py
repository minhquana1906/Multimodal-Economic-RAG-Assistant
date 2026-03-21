import pytest
from unittest.mock import AsyncMock, MagicMock, patch


@pytest.mark.asyncio
async def test_retriever_returns_empty_on_error():
    """Returns empty list when Qdrant is unreachable."""
    from orchestrator.services.retriever import RetrieverClient
    retriever = RetrieverClient("http://qdrant:6333", "test_collection")
    with patch.object(retriever.client, "query_points", new_callable=AsyncMock, side_effect=Exception("connection refused")):
        result = await retriever.hybrid_search(dense_vector=[0.1] * 1024)
    assert result == []


@pytest.mark.asyncio
async def test_retriever_hybrid_passes_sparse_vector():
    """When sparse_vector is provided, query_points is called with Prefetch + FusionQuery."""
    from orchestrator.services.retriever import RetrieverClient
    from qdrant_client.models import FusionQuery, Fusion

    retriever = RetrieverClient("http://qdrant:6333", "test_collection")

    mock_point = MagicMock()
    mock_point.id = "abc123"
    mock_point.score = 0.9
    mock_point.payload = {"text": "test", "source": "s", "title": "t"}

    mock_result = MagicMock()
    mock_result.points = [mock_point]

    with patch.object(retriever.client, "query_points", new_callable=AsyncMock, return_value=mock_result) as mock_qp:
        result = await retriever.hybrid_search(
            dense_vector=[0.1] * 1024,
            sparse_vector={"indices": [1, 42], "values": [0.22, 0.8]},
            top_k=5,
        )

    # Verify the call used Prefetch + FusionQuery (hybrid path)
    call_kwargs = mock_qp.call_args.kwargs
    assert "prefetch" in call_kwargs
    assert len(call_kwargs["prefetch"]) == 2
    assert isinstance(call_kwargs["query"], FusionQuery)
    assert call_kwargs["query"].fusion == Fusion.RRF

    # Verify result shape
    assert len(result) == 1
    assert result[0]["id"] == "abc123"
    assert result[0]["score"] == 0.9


@pytest.mark.asyncio
async def test_retriever_dense_only_uses_named_vector():
    """Dense-only path (sparse_vector=None) uses DENSE_VECTOR_NAME."""
    from orchestrator.services.retriever import RetrieverClient, DENSE_VECTOR_NAME

    retriever = RetrieverClient("http://qdrant:6333", "test_collection")

    mock_point = MagicMock()
    mock_point.id = "xyz"
    mock_point.score = 0.8
    mock_point.payload = {"text": "content", "source": "src", "title": "ttl"}

    mock_result = MagicMock()
    mock_result.points = [mock_point]

    with patch.object(retriever.client, "query_points", new_callable=AsyncMock, return_value=mock_result) as mock_qp:
        result = await retriever.hybrid_search(dense_vector=[0.1] * 1024)

    call_kwargs = mock_qp.call_args.kwargs
    assert call_kwargs.get("using") == DENSE_VECTOR_NAME
    assert "prefetch" not in call_kwargs   # not hybrid
    assert len(result) == 1
    assert result[0]["id"] == "xyz"
