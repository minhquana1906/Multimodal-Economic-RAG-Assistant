from qdrant_client import AsyncQdrantClient
from qdrant_client.models import (
    Prefetch,
    FusionQuery,
    Fusion,
    SparseVector,
)
import logging
from langsmith import traceable

logger = logging.getLogger(__name__)

# Names must match the collection's vector config
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
        """Retrieve relevant chunks using dense-only or hybrid RRF search.

        If sparse_vector is provided ({"indices": [...], "values": [...]}),
        performs hybrid RRF search combining dense + sparse.
        Otherwise, performs dense-only search.

        Returns list of dicts with keys: id, text, source, title, score.
        Returns [] on any error.
        """
        try:
            if sparse_vector is not None:
                # Hybrid RRF: dense + sparse via Prefetch + FusionQuery
                results = await self.client.query_points(
                    collection_name=self.collection,
                    prefetch=[
                        Prefetch(
                            query=dense_vector,
                            using=DENSE_VECTOR_NAME,
                            limit=top_k,
                        ),
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
                # Dense-only fallback
                results = await self.client.query_points(
                    collection_name=self.collection,
                    query=dense_vector,
                    limit=top_k,
                    with_payload=True,
                )
                points = results.points

            return [
                {
                    "id": str(r.id),
                    "text": r.payload.get("text", ""),
                    "source": r.payload.get("source", ""),
                    "title": r.payload.get("title", ""),
                    "score": r.score,
                }
                for r in points
            ]
        except Exception as e:
            logger.error(f"Retrieval error: {e}")
            return []
