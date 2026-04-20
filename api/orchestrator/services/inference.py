from __future__ import annotations

import httpx
from langsmith import traceable
from loguru import logger


class InferenceClient:
    """Client for the consolidated inference service (embed, sparse, rerank)."""

    def __init__(self, base_url: str, timeout: float = 30.0):
        self.base_url = base_url.rstrip("/")
        self.timeout = timeout

    @traceable(name="Embed Query", run_type="chain")
    async def embed_query(self, text: str) -> list[float]:
        """Embed a single query text. Returns the first embedding vector."""
        async with httpx.AsyncClient() as client:
            response = await client.post(
                f"{self.base_url}/embed",
                json={"texts": [text]},
                timeout=self.timeout,
            )
            response.raise_for_status()
            return response.json()["embeddings"][0]

    @traceable(name="Embed Documents", run_type="chain")
    async def embed_documents(self, texts: list[str]) -> list[list[float]]:
        """Embed multiple document texts. Returns all embedding vectors."""
        async with httpx.AsyncClient() as client:
            response = await client.post(
                f"{self.base_url}/embed",
                json={"texts": texts},
                timeout=self.timeout,
            )
            response.raise_for_status()
            return response.json()["embeddings"]

    @traceable(name="Sparse Query", run_type="chain")
    async def sparse_query(self, text: str) -> dict:
        """Generate sparse vector for a single query. Returns {indices, values}."""
        async with httpx.AsyncClient() as client:
            response = await client.post(
                f"{self.base_url}/sparse",
                json={"texts": [text]},
                timeout=self.timeout,
            )
            response.raise_for_status()
            return response.json()["vectors"][0]

    @traceable(name="Sparse Documents", run_type="chain")
    async def sparse_documents(self, texts: list[str]) -> list[dict]:
        """Generate sparse vectors for a batch of texts. Returns all vectors."""
        async with httpx.AsyncClient() as client:
            response = await client.post(
                f"{self.base_url}/sparse",
                json={"texts": texts},
                timeout=self.timeout,
            )
            response.raise_for_status()
            return response.json()["vectors"]

    @traceable(name="Rerank Passages", run_type="chain")
    async def rerank(self, query: str, passages: list[str]) -> list[float]:
        """Rerank passages by relevance to query. Returns scores list (normalized 0-1)."""
        try:
            async with httpx.AsyncClient() as client:
                response = await client.post(
                    f"{self.base_url}/rerank",
                    json={"query": query, "passages": passages},
                    timeout=self.timeout,
                )
                response.raise_for_status()
                return response.json()["scores"]
        except Exception as e:
            logger.error(f"Reranking error: {e}")
            return [0.0] * len(passages)
