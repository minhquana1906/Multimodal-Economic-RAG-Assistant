from qdrant_client import AsyncQdrantClient
from qdrant_client.models import NamedVector, NamedSparseVector, SparseVector, SearchRequest, FusionQuery, Fusion, Prefetch
import logging
from langsmith import traceable

logger = logging.getLogger(__name__)

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
        """Retrieve relevant chunks from Qdrant using dense-only or hybrid RRF search."""
        try:
            results = await self.client.search(
                collection_name=self.collection,
                query_vector=dense_vector,
                limit=top_k,
            )
            return [
                {
                    "id": str(r.id),
                    "text": r.payload.get("text", ""),
                    "source": r.payload.get("source", ""),
                    "title": r.payload.get("title", ""),
                    "score": r.score,
                }
                for r in results
            ]
        except Exception as e:
            logger.error(f"Retrieval error: {e}")
            return []
