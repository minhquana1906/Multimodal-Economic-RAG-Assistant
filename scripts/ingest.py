"""
scripts/ingest.py — Main data ingestion pipeline.

Loads the khoalnd/EconVNNews dataset from HuggingFace, chunks articles
semantically, generates dense (embedding service) + sparse (BM25 fastembed)
vectors, and upserts them into a Qdrant collection.

Idempotent: if the collection already holds ≥ EXPECTED_MIN points the script
exits early without re-ingesting.
"""

from __future__ import annotations

import asyncio
import os
import sys
from typing import Any

from loguru import logger

import httpx
from datasets import load_dataset
from fastembed.sparse.bm25 import Bm25
from qdrant_client import QdrantClient
from qdrant_client.models import (
    Distance,
    NamedSparseVector,
    NamedVector,
    PointStruct,
    SparseIndexParams,
    SparseVectorParams,
    VectorParams,
)
from underthesea import word_tokenize

from chunker import chunk_article, make_chunk_id

try:
    from langsmith import traceable
except ImportError:  # LangSmith is optional at import time

    def traceable(name: str = ""):  # type: ignore[misc]
        def decorator(fn):
            return fn

        return decorator


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

logger.remove()
logger.add(sys.stderr, format="{time:HH:mm:ss} | {level} | {message}", level="INFO")

os.environ.setdefault("LANGSMITH_PROJECT", "multimodal-economic-rag-ingest")

DATASET_NAME: str = "khoalnd/EconVNNews"
EMBEDDING_URL: str = os.getenv("EMBEDDING_URL", "http://embedding:8001")
QDRANT_URL: str = os.getenv("QDRANT_URL", "http://qdrant:6333")
COLLECTION_NAME: str = "econ_vn_news"
BATCH_SIZE: int = int(os.getenv("INGEST_BATCH_SIZE", "256"))
DENSE_DIM: int = 1024
MAX_EMBED_RETRIES: int = 3
EMBED_RETRY_DELAY: float = 5.0
EXPECTED_MIN: int = 400_000

# ---------------------------------------------------------------------------
# Utilities
# ---------------------------------------------------------------------------


def tokenize_vietnamese(text: str) -> str:
    """Segment Vietnamese text into space-separated tokens using underthesea."""
    return word_tokenize(text, format="text")


# ---------------------------------------------------------------------------
# Qdrant collection helpers
# ---------------------------------------------------------------------------


def create_collection(
    client: QdrantClient, collection_name: str, dense_dim: int = DENSE_DIM
) -> None:
    """Create a Qdrant collection with dense (COSINE) + sparse (BM25) vectors."""
    client.create_collection(
        collection_name=collection_name,
        vectors_config={
            "dense": VectorParams(
                size=dense_dim,
                distance=Distance.COSINE,
            ),
        },
        sparse_vectors_config={
            "sparse": SparseVectorParams(
                index=SparseIndexParams(on_disk=False),
            ),
        },
    )
    logger.info(
        f"Created collection '{collection_name}' (dense={dense_dim}, sparse=BM25)"
    )


def should_skip_ingestion(
    client: QdrantClient, collection_name: str, expected_min: int = EXPECTED_MIN
) -> bool:
    """Return True if the collection already has enough points to skip re-ingestion."""
    try:
        info = client.get_collection(collection_name)
        count = info.points_count or 0
        if count >= expected_min:
            logger.info(
                f"Collection '{collection_name}' already has {count} >= {expected_min} points; skipping ingestion."
            )
            return True
    except Exception:
        pass  # Collection doesn't exist yet
    return False


# ---------------------------------------------------------------------------
# Embedding service client
# ---------------------------------------------------------------------------


@traceable(name="Get Dense Embeddings Batch")
async def get_dense_embeddings(
    texts: list[str], is_query: bool = False
) -> list[list[float]]:
    """Fetch dense embeddings from the embedding service with OOM retry."""
    async with httpx.AsyncClient(timeout=300.0) as client:
        for attempt in range(1, MAX_EMBED_RETRIES + 1):
            response = await client.post(
                f"{EMBEDDING_URL}/embed",
                json={"texts": texts, "is_query": is_query},
            )
            if response.status_code == 500 and attempt < MAX_EMBED_RETRIES:
                logger.warning(
                    f"Embedding request failed (attempt {attempt}/{MAX_EMBED_RETRIES}), "
                    f"retrying in {EMBED_RETRY_DELAY}s..."
                )
                await asyncio.sleep(EMBED_RETRY_DELAY)
                continue
            response.raise_for_status()
            return response.json()["embeddings"]


# ---------------------------------------------------------------------------
# Pipeline steps
# ---------------------------------------------------------------------------


@traceable(name="Load and Chunk Articles")
def load_and_chunk_articles() -> list[dict]:
    """Load EconVNNews dataset and chunk all articles."""
    logger.info(f"Loading dataset '{DATASET_NAME}'...")
    dataset = load_dataset(DATASET_NAME, split="train", streaming=False)

    all_chunks: list[dict] = []
    for i, article in enumerate(dataset):
        chunks = chunk_article(article)
        all_chunks.extend(chunks)
        if (i + 1) % 10_000 == 0:
            logger.info(f"Chunked {i+1} articles → {len(all_chunks)} chunks so far")

    logger.info(f"Total: {len(all_chunks)} chunks from {len(dataset)} articles")
    return all_chunks


def _build_sparse_vectors(texts: list[str], bm25: Bm25) -> list[dict[str, Any]]:
    """Generate BM25 sparse vectors for a batch of (Vietnamese-tokenised) texts."""
    tokenised = [tokenize_vietnamese(t) for t in texts]
    embeddings = list(bm25.embed(tokenised))  # yields SparseEmbedding objects
    return [
        {"indices": list(map(int, e.indices)), "values": list(map(float, e.values))}
        for e in embeddings
    ]


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


@traceable(name="Data Ingestion Pipeline")
async def main() -> None:
    logger.info("Starting data ingestion pipeline...")

    qdrant = QdrantClient(url=QDRANT_URL)

    # Idempotency guard
    if should_skip_ingestion(qdrant, COLLECTION_NAME):
        return

    # Ensure collection exists
    existing = [c.name for c in qdrant.get_collections().collections]
    if COLLECTION_NAME not in existing:
        create_collection(qdrant, COLLECTION_NAME, dense_dim=DENSE_DIM)

    # Load & chunk
    chunks = load_and_chunk_articles()

    # Initialise BM25 sparse encoder (fastembed)
    bm25 = Bm25("Qdrant/bm25")

    total = len(chunks)
    points_upserted = 0

    for batch_start in range(0, total, BATCH_SIZE):
        batch = chunks[batch_start : batch_start + BATCH_SIZE]
        texts = [c["text"] for c in batch]

        # Dense embeddings
        dense_vecs: list[list[float]] = await get_dense_embeddings(
            texts, is_query=False
        )

        # Sparse BM25 vectors
        sparse_vecs = _build_sparse_vectors(texts, bm25)

        # Build Qdrant points
        qdrant_points: list[PointStruct] = []
        for chunk, dense, sparse in zip(batch, dense_vecs, sparse_vecs):
            pid = make_chunk_id(chunk["url"], chunk["chunk_index"])
            qdrant_points.append(
                PointStruct(
                    id=pid,
                    vector={
                        "dense": dense,
                        "sparse": sparse,
                    },
                    payload={
                        "text": chunk["text"],
                        "title": chunk["title"],
                        "url": chunk["url"],
                        "source": chunk.get("source", ""),
                        "chunk_type": chunk["chunk_type"],
                        "chunk_index": chunk["chunk_index"],
                        "published_date": chunk.get("published_date", ""),
                        "category": chunk.get("category", ""),
                    },
                )
            )

        qdrant.upsert(collection_name=COLLECTION_NAME, points=qdrant_points)
        points_upserted += len(qdrant_points)

        batch_num = batch_start // BATCH_SIZE + 1
        if batch_num % 10 == 0 or batch_start + BATCH_SIZE >= total:
            logger.info(
                f"Batch {batch_num}: upserted {points_upserted} / {total} chunks"
            )

    logger.info(
        f"Ingestion complete! {points_upserted} points upserted to '{COLLECTION_NAME}'."
    )


if __name__ == "__main__":
    asyncio.run(main())
