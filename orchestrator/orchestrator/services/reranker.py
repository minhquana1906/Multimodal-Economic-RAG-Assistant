import httpx
import logging
from langsmith import traceable

logger = logging.getLogger(__name__)

class RerankerClient:
    def __init__(self, url: str, timeout: float):
        self.url = url
        self.timeout = timeout

    @traceable(name="Rerank Passages", run_type="chain")
    async def rerank(self, query: str, passages: list[str], top_n: int = 5) -> list[dict]:
        """Rerank passages by relevance. Returns list of {index, score} sorted desc.
        Falls back to original order if service unavailable."""
        try:
            async with httpx.AsyncClient() as client:
                response = await client.post(
                    f"{self.url}/rerank",
                    json={"query": query, "passages": passages},
                    timeout=self.timeout,
                )
                response.raise_for_status()
                scores = response.json()["scores"]
                ranked = sorted(
                    [{"index": i, "score": s} for i, s in enumerate(scores)],
                    key=lambda x: x["score"],
                    reverse=True,
                )
                return ranked[:top_n]
        except Exception as e:
            logger.error(f"Reranking error: {e}")
            # Fallback: return original order with score=0
            return [{"index": i, "score": 0.0} for i in range(min(len(passages), top_n))]
