import pytest
from unittest.mock import AsyncMock, MagicMock, patch


@pytest.mark.asyncio
async def test_web_search_returns_empty_without_api_key():
    """Returns empty list gracefully when no API key is configured."""
    from orchestrator.services.web_search import WebSearchClient

    client = WebSearchClient(api_key="")
    result = await client.search("GDP Vietnam")
    assert result == []


@pytest.mark.asyncio
async def test_web_search_normalizes_url_source_and_score():
    """Tavily results are normalized into canonical citation fields."""
    from orchestrator.services.web_search import WebSearchClient

    client = WebSearchClient(api_key="tvly-test")

    mock_response = MagicMock()
    mock_response.raise_for_status = MagicMock()
    mock_response.json.return_value = {
        "results": [
            {
                "title": "GDP outlook",
                "url": "https://example.com/economy/gdp-outlook",
                "content": "GDP content",
                "score": 0.42,
            }
        ]
    }

    mock_client = AsyncMock()
    mock_client.post = AsyncMock(return_value=mock_response)

    mock_async_cm = MagicMock()
    mock_async_cm.__aenter__ = AsyncMock(return_value=mock_client)
    mock_async_cm.__aexit__ = AsyncMock(return_value=False)

    with patch("httpx.AsyncClient", return_value=mock_async_cm):
        result = await client.search("GDP Vietnam")

    assert result == [
        {
            "text": "GDP content",
            "title": "GDP outlook",
            "url": "https://example.com/economy/gdp-outlook",
            "source": "example.com",
            "score": 0.42,
        }
    ]
