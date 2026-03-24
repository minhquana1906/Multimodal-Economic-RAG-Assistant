from __future__ import annotations

from urllib.parse import urlparse

import httpx
from langsmith import traceable
from loguru import logger


class WebSearchClient:
    def __init__(self, api_key: str = "", timeout: float = 10.0):
        self.api_key = api_key
        self.timeout = timeout

    def _source_from_url(self, url: str) -> str:
        if not url:
            return ""
        return urlparse(url).netloc or url

    @traceable(name="Web Search Fallback", run_type="retriever")
    async def search(self, query: str, max_results: int = 5) -> list[dict]:
        """Search web via Tavily API. Returns [] on error or missing API key."""
        if not self.api_key:
            logger.info("Web search API key not set; skipping web search")
            return []
        try:
            async with httpx.AsyncClient() as client:
                response = await client.post(
                    "https://api.tavily.com/search",
                    json={
                        "api_key": self.api_key,
                        "query": query,
                        "max_results": max_results,
                    },
                    timeout=self.timeout,
                )
                response.raise_for_status()
                results = response.json().get("results", [])
                return [
                    {
                        "text": r.get("content", ""),
                        "title": r.get("title", ""),
                        "url": r.get("url", ""),
                        "source": self._source_from_url(r.get("url", "")),
                        "score": r.get("score", 0.0),
                    }
                    for r in results
                ]
        except Exception as e:
            logger.error(f"Web search error: {e}")
            return []
