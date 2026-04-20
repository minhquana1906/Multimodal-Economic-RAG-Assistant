"""
scripts/ingest.py — Main data ingestion pipeline.

Loads the khoalnd/EconVNNews dataset from HuggingFace, chunks articles
semantically, generates dense + sparse vectors via the inference service,
and upserts them into a Qdrant collection.

Idempotent: if the collection already holds ≥ EXPECTED_MIN points the script
exits early without re-ingesting.
"""

from __future__ import annotations

import asyncio
import os
import sys
from typing import Any

import httpx
from datasets import load_dataset
from loguru import logger
from qdrant_client import QdrantClient
from qdrant_client.models import (
    Distance,
    PointStruct,
    SparseIndexParams,
    SparseVectorParams,
    VectorParams,
)

from chunker import (
    MAX_CHUNK_CHARS,
    MAX_CHUNK_WORDS,
    chunk_article,
    make_chunk_id,
)

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
INFERENCE_URL: str = os.getenv("SERVICES__INFERENCE_URL", "http://inference:8001")
QDRANT_URL: str = os.getenv("SERVICES__QDRANT_URL", "http://qdrant:6333")
COLLECTION_NAME: str = os.getenv("SERVICES__QDRANT_COLLECTION", "econ_vn_news")
BATCH_SIZE: int = int(os.getenv("INGEST__BATCH_SIZE", "256"))
DENSE_DIM: int = 1024
MAX_EMBED_RETRIES: int = 3
EMBED_RETRY_DELAY: float = 5.0
EXPECTED_MIN: int = 400_000
CHUNKING_VERSION: int = 2
FORCE_RECREATE: bool = (
    os.getenv("INGEST__FORCE_RECREATE", "false").strip().lower()
    in {"1", "true", "yes", "on"}
)

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


def get_collection_chunking_version(
    client: QdrantClient, collection_name: str
) -> tuple[bool, int | None]:
    """Return whether the collection has points and which chunking version they use."""
    records, _ = client.scroll(
        collection_name=collection_name,
        limit=1,
        with_payload=True,
        with_vectors=False,
    )
    if not records:
        return False, None

    payload = getattr(records[0], "payload", {}) or {}
    version = payload.get("chunking_version")
    try:
        return True, int(version) if version is not None else None
    except (TypeError, ValueError):
        return True, None


def ensure_collection_ready(
    client: QdrantClient,
    collection_name: str,
    dense_dim: int = DENSE_DIM,
    force_recreate: bool = FORCE_RECREATE,
) -> None:
    """Ensure the target collection is compatible with the current chunking strategy."""
    existing = [c.name for c in client.get_collections().collections]
    if collection_name not in existing:
        create_collection(client, collection_name, dense_dim=dense_dim)
        return

    if force_recreate:
        logger.warning(
            f"Recreating collection '{collection_name}' because INGEST__FORCE_RECREATE is enabled."
        )
        client.delete_collection(collection_name=collection_name)
        create_collection(client, collection_name, dense_dim=dense_dim)
        return

    has_points, version = get_collection_chunking_version(client, collection_name)
    if has_points and version != CHUNKING_VERSION:
        raise RuntimeError(
            f"Collection '{collection_name}' uses chunking version {version!r}, "
            f"expected {CHUNKING_VERSION}. Re-run with INGEST__FORCE_RECREATE=true "
            "or ingest into a new collection name to avoid mixing old long chunks "
            "with the new bounded chunking strategy."
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


def validate_chunk_batch(batch: list[dict]) -> None:
    """Fail fast if any chunk still violates the ingestion size budget."""
    oversized: list[str] = []

    for chunk in batch:
        text = chunk.get("text", "")
        word_count = len(text.split())
        if len(text) <= MAX_CHUNK_CHARS and word_count <= MAX_CHUNK_WORDS:
            continue

        oversized.append(
            f"url={chunk.get('url', '')} chunk_index={chunk.get('chunk_index', '?')} "
            f"chars={len(text)} words={word_count}"
        )

    if oversized:
        details = "; ".join(oversized[:3])
        raise ValueError(
            "Oversized chunk detected before embedding/upsert. "
            f"Budget chars<={MAX_CHUNK_CHARS}, words<={MAX_CHUNK_WORDS}. "
            f"Examples: {details}"
        )


# ---------------------------------------------------------------------------
# Inference service client helpers
# ---------------------------------------------------------------------------


async def _embed_with_retry(client: httpx.AsyncClient, texts: list[str]) -> list[list[float]]:
    """Fetch dense embeddings from the inference service with OOM retry."""
    for attempt in range(1, MAX_EMBED_RETRIES + 1):
        response = await client.post(
            f"{INFERENCE_URL}/embed",
            json={"texts": texts},
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


async def _sparse_with_retry(client: httpx.AsyncClient, texts: list[str]) -> list[dict[str, Any]]:
    """Fetch sparse vectors from the inference service with OOM retry."""
    for attempt in range(1, MAX_EMBED_RETRIES + 1):
        response = await client.post(
            f"{INFERENCE_URL}/sparse",
            json={"texts": texts},
        )
        if response.status_code == 500 and attempt < MAX_EMBED_RETRIES:
            logger.warning(
                f"Sparse request failed (attempt {attempt}/{MAX_EMBED_RETRIES}), "
                f"retrying in {EMBED_RETRY_DELAY}s..."
            )
            await asyncio.sleep(EMBED_RETRY_DELAY)
            continue
        response.raise_for_status()
        return response.json()["vectors"]


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


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


@traceable(name="Data Ingestion Pipeline")
async def main() -> None:
    logger.info("Starting data ingestion pipeline...")

    qdrant = QdrantClient(url=QDRANT_URL)

    ensure_collection_ready(
        qdrant,
        COLLECTION_NAME,
        dense_dim=DENSE_DIM,
        force_recreate=FORCE_RECREATE,
    )

    # Idempotency guard
    if should_skip_ingestion(qdrant, COLLECTION_NAME):
        return

    # Load & chunk
    chunks = load_and_chunk_articles()

    total = len(chunks)
    points_upserted = 0

    async with httpx.AsyncClient(timeout=300.0) as http_client:
        for batch_start in range(0, total, BATCH_SIZE):
            batch = chunks[batch_start : batch_start + BATCH_SIZE]
            validate_chunk_batch(batch)
            texts = [c["text"] for c in batch]

            # Dense and sparse vectors via inference service
            dense_vecs: list[list[float]] = await _embed_with_retry(http_client, texts)
            sparse_vecs = await _sparse_with_retry(http_client, texts)

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
                            "chunking_version": CHUNKING_VERSION,
                            "char_count": len(chunk["text"]),
                            "word_count": len(chunk["text"].split()),
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
