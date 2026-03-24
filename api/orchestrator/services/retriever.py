from __future__ import annotations

from langsmith import traceable
from loguru import logger
from qdrant_client import AsyncQdrantClient
from qdrant_client.models import Fusion, FusionQuery, Prefetch, SparseVector

DENSE_VECTOR_NAME = "dense"
SPARSE_VECTOR_NAME = "sparse"


class RetrieverClient:
    def __init__(self, url: str, collection: str):
        self.client = AsyncQdrantClient(url=url)
        self.collection = collection

    @traceable(name="Hybrid Retrieval", run_type="retriever")
    async def hybrid_search(
        self,
        dense_vector: list[float],
        sparse_vector: dict | None = None,
        top_k: int = 20,
    ) -> list[dict]:
        """Retrieve relevant chunks using dense-only or hybrid RRF search."""
        try:
            if sparse_vector is not None:
                results = await self.client.query_points(
                    collection_name=self.collection,
                    prefetch=[
                        Prefetch(query=dense_vector, using=DENSE_VECTOR_NAME, limit=top_k),
                        Prefetch(
                            query=SparseVector(
                                indices=sparse_vector["indices"],
                                values=sparse_vector["values"],
                            ),
                            using=SPARSE_VECTOR_NAME,
                            limit=top_k,
                        ),
                    ],
                    query=FusionQuery(fusion=Fusion.RRF),
                    limit=top_k,
                    with_payload=True,
                )
                points = results.points
            else:
                results = await self.client.query_points(
                    collection_name=self.collection,
                    query=dense_vector,
                    using=DENSE_VECTOR_NAME,
                    limit=top_k,
                    with_payload=True,
                )
                points = results.points

            docs = [
                {
                    "id": str(r.id),
                    "text": r.payload.get("text", ""),
                    "source": r.payload.get("source", ""),
                    "title": r.payload.get("title", ""),
                    "url": r.payload.get("url", ""),
                    "score": r.score,
                }
                for r in points
            ]
            logger.log(
                "RETRIEVAL",
                f"top_k={top_k} hits={len(docs)} sparse={sparse_vector is not None}",
            )
            return docs
        except Exception as e:
            logger.error(f"Retrieval error: {e}")
            return []
