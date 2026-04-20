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


@pytest.mark.asyncio
async def test_rerank_sends_instruction_in_payload():
    """instruction is always present in the HTTP request body."""
    from orchestrator.services.reranker import RerankerClient
    from unittest.mock import AsyncMock, MagicMock, patch

    mock_response = MagicMock()
    mock_response.raise_for_status = MagicMock()
    mock_response.json = MagicMock(return_value={"scores": [0.9, 0.5]})

    posted_json = {}

    async def fake_post(url, json=None, timeout=None):
        posted_json.update(json or {})
        return mock_response

    client = RerankerClient(url="http://test:8002", timeout=5.0)
    with patch("httpx.AsyncClient.post", new=AsyncMock(side_effect=fake_post)):
        await client.rerank("query", ["p1", "p2"], top_n=2, instruction="test inst")

    assert "instruction" in posted_json
    assert posted_json["instruction"] == "test inst"


@pytest.mark.asyncio
async def test_rerank_instruction_none_still_in_payload():
    """When instruction=None (default), key is still present in payload as None."""
    from orchestrator.services.reranker import RerankerClient
    from unittest.mock import AsyncMock, MagicMock, patch

    mock_response = MagicMock()
    mock_response.raise_for_status = MagicMock()
    mock_response.json = MagicMock(return_value={"scores": [0.9]})

    posted_json = {}

    async def fake_post(url, json=None, timeout=None):
        posted_json.update(json or {})
        return mock_response

    client = RerankerClient(url="http://test:8002", timeout=5.0)
    with patch("httpx.AsyncClient.post", new=AsyncMock(side_effect=fake_post)):
        await client.rerank("query", ["p1"], top_n=1)

    assert "instruction" in posted_json
    assert posted_json["instruction"] is None
