"""Bootstrap Qdrant collection and payload indexes for the academic_chunks schema."""

import os

from loguru import logger
from qdrant_client import QdrantClient
from qdrant_client.http import models as qm

QDRANT_URL = os.environ.get("SERVICES__QDRANT_URL", "http://localhost:6333")
COLLECTION = os.environ.get("SERVICES__QDRANT_COLLECTION", "academic_chunks")

INDEXED_FIELDS: list[tuple[str, qm.PayloadSchemaType]] = [
    ("source", qm.PayloadSchemaType.KEYWORD),
    ("doc_type", qm.PayloadSchemaType.KEYWORD),
    ("category", qm.PayloadSchemaType.KEYWORD),
    ("language", qm.PayloadSchemaType.KEYWORD),
    ("term_vi", qm.PayloadSchemaType.KEYWORD),
    ("term_en", qm.PayloadSchemaType.KEYWORD),
]


def bootstrap() -> None:
    """Create collection and payload indexes if missing."""
    client = QdrantClient(url=QDRANT_URL, prefer_grpc=False)

    existing = {c.name for c in client.get_collections().collections}
    if COLLECTION not in existing:
        client.create_collection(
            collection_name=COLLECTION,
            vectors_config={
                "dense": qm.VectorParams(size=1024, distance=qm.Distance.COSINE),
            },
            sparse_vectors_config={
                "sparse": qm.SparseVectorParams(modifier=qm.Modifier.IDF),
            },
        )
        logger.info(f"created collection {COLLECTION}")
    else:
        logger.info(f"collection {COLLECTION} already exists")

    for field, schema in INDEXED_FIELDS:
        client.create_payload_index(
            collection_name=COLLECTION,
            field_name=field,
            field_schema=schema,
        )
        logger.info(f"indexed payload field {field}")


if __name__ == "__main__":
    bootstrap()
