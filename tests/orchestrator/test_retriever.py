import pytest
from unittest.mock import AsyncMock, patch


@pytest.mark.asyncio
async def test_retriever_returns_empty_on_error():
    """Returns empty list when Qdrant is unreachable."""
    from orchestrator.services.retriever import RetrieverClient
    retriever = RetrieverClient("http://qdrant:6333", "test_collection")
    # Patch the internal qdrant client's search to raise
    with patch.object(retriever.client, "search", new_callable=AsyncMock, side_effect=Exception("connection refused")):
        result = await retriever.hybrid_search(dense_vector=[0.1] * 1024)
    assert result == []
