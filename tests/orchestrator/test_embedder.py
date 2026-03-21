import pytest
from unittest.mock import AsyncMock, MagicMock, patch


@pytest.mark.asyncio
async def test_embedder_returns_vector():
    """embed_query returns a 1024-dim float list."""
    from orchestrator.services.embedder import EmbedderClient
    embedder = EmbedderClient("http://embedding:8001", timeout=15.0)

    mock_response = MagicMock()
    mock_response.json.return_value = {"embeddings": [[0.1] * 1024]}
    mock_response.raise_for_status = MagicMock()

    mock_client = AsyncMock()
    mock_client.post = AsyncMock(return_value=mock_response)

    mock_async_cm = MagicMock()
    mock_async_cm.__aenter__ = AsyncMock(return_value=mock_client)
    mock_async_cm.__aexit__ = AsyncMock(return_value=False)

    with patch("httpx.AsyncClient", return_value=mock_async_cm):
        result = await embedder.embed_query("GDP Việt Nam")

    assert len(result) == 1024
    assert isinstance(result[0], float)
