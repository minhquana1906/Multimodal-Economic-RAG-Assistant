import httpx
import logging
from langsmith import traceable

logger = logging.getLogger(__name__)

class WebSearchClient:
    def __init__(self, api_key: str = ""):
        self.api_key = api_key

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
                    json={"api_key": self.api_key, "query": query, "max_results": max_results},
                    timeout=10.0,
                )
                response.raise_for_status()
                results = response.json().get("results", [])
                return [
                    {"text": r.get("content", ""), "source": r.get("url", ""), "title": r.get("title", "")}
                    for r in results
                ]
        except Exception as e:
            logger.error(f"Web search error: {e}")
            return []
