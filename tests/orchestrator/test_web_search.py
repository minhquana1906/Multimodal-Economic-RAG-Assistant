import pytest


@pytest.mark.asyncio
async def test_web_search_returns_empty_without_api_key():
    """Returns empty list gracefully when no API key is configured."""
    from orchestrator.services.web_search import WebSearchClient
    client = WebSearchClient(api_key="")
    result = await client.search("GDP Vietnam")
    assert result == []
