from __future__ import annotations

import httpx
from langsmith import traceable
from loguru import logger


class RerankerClient:
    def __init__(self, url: str, timeout: float):
        self.url = url
        self.timeout = timeout

    @traceable(name="Rerank Passages", run_type="chain")
    async def rerank(
        self,
        query: str,
        passages: list[str],
        top_n: int = 5,
        instruction: str | None = None,
    ) -> list[dict]:
        """Rerank passages by relevance. Returns list of {index, score} sorted desc.
        Falls back to original order if service unavailable."""
        try:
            payload = {
                "query": query,
                "passages": passages,
                "instruction": instruction,
            }
            async with httpx.AsyncClient() as client:
                response = await client.post(
                    f"{self.url}/rerank",
                    json=payload,
                    timeout=self.timeout,
                )
                response.raise_for_status()
                scores = response.json()["scores"]
                ranked = sorted(
                    [{"index": i, "score": s} for i, s in enumerate(scores)],
                    key=lambda x: x["score"],
                    reverse=True,
                )
                result = ranked[:top_n]
                top_score = result[0]["score"] if result else 0.0
                logger.log(
                    "RERANK",
                    "input={} output={} top_score={:.4f}",
                    len(passages),
                    len(result),
                    top_score,
                )
                return result
        except Exception as e:
            logger.error("Reranking error: {}", e)
            return [{"index": i, "score": 0.0} for i in range(min(len(passages), top_n))]
