import pytest
from unittest.mock import AsyncMock, MagicMock, patch


@pytest.mark.asyncio
async def test_reranker_falls_back_to_original_order():
    """Falls back to original order (score=0) when service unreachable."""
    from orchestrator.services.reranker import RerankerClient
    reranker = RerankerClient("http://localhost:9999", timeout=0.1)
    result = await reranker.rerank("test query", ["doc1", "doc2", "doc3"], top_n=3)
    assert len(result) == 3
    assert all(r["score"] == 0.0 for r in result)
    assert result[0]["index"] == 0
