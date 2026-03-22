from __future__ import annotations

import httpx
from langsmith import traceable
from loguru import logger


class EmbedderClient:
    def __init__(self, url: str, timeout: float):
        self.url = url
        self.timeout = timeout

    @traceable(name="Embed Query", run_type="chain")
    async def embed_query(self, text: str) -> list[float]:
        """Embed a single query text. Returns 1024-dim vector. Raises on error."""
        try:
            async with httpx.AsyncClient() as client:
                response = await client.post(
                    f"{self.url}/embed",
                    json={"texts": [text], "is_query": True},
                    timeout=self.timeout,
                )
                response.raise_for_status()
                return response.json()["embeddings"][0]
        except Exception as e:
            logger.error("Embedding error: {}", e)
            raise

    @traceable(name="Embed Documents", run_type="chain")
    async def embed_documents(self, texts: list[str]) -> list[list[float]]:
        """Embed multiple document texts. Returns list of 1024-dim vectors. Raises on error."""
        try:
            async with httpx.AsyncClient() as client:
                response = await client.post(
                    f"{self.url}/embed",
                    json={"texts": texts, "is_query": False},
                    timeout=self.timeout,
                )
                response.raise_for_status()
                return response.json()["embeddings"]
        except Exception as e:
            logger.error("Embedding error: {}", e)
            raise
