from __future__ import annotations

from fastembed.sparse.bm25 import Bm25
from loguru import logger
from underthesea import word_tokenize


def tokenize_vietnamese(text: str) -> str:
    """Segment Vietnamese text into space-separated tokens using underthesea."""
    return word_tokenize(text, format="text")


class SparseEncoderService:
    def __init__(self, model_name: str = "Qdrant/bm25"):
        self.model_name = model_name
        self._bm25: Bm25 | None = None
        self._initialization_error: Exception | None = None

    def _get_bm25(self) -> Bm25:
        if self._bm25 is not None:
            return self._bm25
        if self._initialization_error is not None:
            raise self._initialization_error

        try:
            self._bm25 = Bm25(self.model_name)
            return self._bm25
        except Exception as exc:
            self._initialization_error = exc
            logger.error(f"Sparse encoder initialization failed: {exc}")
            raise

    def encode_query(self, text: str) -> dict[str, list[float] | list[int]]:
        tokenized = tokenize_vietnamese(text)
        embedding = next(iter(self._get_bm25().query_embed(tokenized)))
        return {
            "indices": list(map(int, embedding.indices)),
            "values": list(map(float, embedding.values)),
        }
