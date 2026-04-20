import asyncio
import math
from contextlib import asynccontextmanager
from typing import Any

from fastapi import FastAPI
from loguru import logger
from pydantic import BaseModel, Field
from starlette.responses import JSONResponse

from FlagEmbedding import BGEM3FlagModel, FlagReranker


EMBEDDER_MODEL_NAME = "BAAI/bge-m3"
RERANKER_MODEL_NAME = "BAAI/bge-reranker-v2-m3"

embedder: BGEM3FlagModel | None = None
reranker: FlagReranker | None = None


class EmbedRequest(BaseModel):
    texts: list[str] = Field(..., min_length=1)


class EmbedResponse(BaseModel):
    embeddings: list[list[float]]


class SparseResponseItem(BaseModel):
    indices: list[int]
    values: list[float]


class SparseResponse(BaseModel):
    vectors: list[SparseResponseItem]


class RerankRequest(BaseModel):
    query: str
    passages: list[str] = Field(..., min_length=1)


class RerankResponse(BaseModel):
    scores: list[float]


def _ensure_loaded() -> tuple[BGEM3FlagModel, FlagReranker]:
    if embedder is None or reranker is None:
        raise RuntimeError("Model is still loading")
    return embedder, reranker


def _sigmoid(x: float) -> float:
    if x >= 0:
        z = math.exp(-x)
        return 1.0 / (1.0 + z)
    z = math.exp(x)
    return z / (1.0 + z)


def _normalize_scores(scores: list[float]) -> list[float]:
    # Convert arbitrary reranker scores to (0, 1) in a stable way.
    return [_sigmoid(float(s)) for s in scores]


def _encode_dense_and_sparse(texts: list[str]) -> tuple[list[list[float]], list[dict[int, float]]]:
    enc: dict[str, Any] = embedder.encode(texts, return_dense=True, return_sparse=True)  # type: ignore[union-attr]
    dense_vecs = enc.get("dense_vecs")
    lexical_weights = enc.get("lexical_weights")
    if dense_vecs is None or lexical_weights is None:
        raise RuntimeError("Embedder did not return dense_vecs and lexical_weights")
    return dense_vecs, lexical_weights


@asynccontextmanager
async def lifespan(_app: FastAPI):
    global embedder, reranker
    embedder = await asyncio.to_thread(BGEM3FlagModel, EMBEDDER_MODEL_NAME, use_fp16=True)
    reranker = await asyncio.to_thread(FlagReranker, RERANKER_MODEL_NAME, use_fp16=True)
    logger.info(
        f"Inference models loaded: embedder={EMBEDDER_MODEL_NAME} reranker={RERANKER_MODEL_NAME}"
    )
    yield
    embedder = None
    reranker = None


app = FastAPI(title="inference", lifespan=lifespan)


@app.get("/health")
async def health():
    if embedder is None or reranker is None:
        return JSONResponse({"status": "loading"}, status_code=503)
    return JSONResponse(
        {
            "status": "ok",
            "embedding_model": getattr(embedder, "model_name", EMBEDDER_MODEL_NAME),
            "reranker_model": getattr(reranker, "model_name", RERANKER_MODEL_NAME),
        }
    )


@app.post("/embed", response_model=EmbedResponse)
async def embed(request: EmbedRequest):
    try:
        _ensure_loaded()
    except RuntimeError:
        return JSONResponse({"detail": "Model is still loading"}, status_code=503)
    dense_vecs, _ = await asyncio.to_thread(_encode_dense_and_sparse, request.texts)
    return EmbedResponse(embeddings=dense_vecs)


@app.post("/sparse", response_model=SparseResponse)
async def sparse(request: EmbedRequest):
    try:
        _ensure_loaded()
    except RuntimeError:
        return JSONResponse({"detail": "Model is still loading"}, status_code=503)
    _, lexical_weights = await asyncio.to_thread(_encode_dense_and_sparse, request.texts)

    items: list[SparseResponseItem] = []
    for weights in lexical_weights:
        indices = sorted(int(k) for k in weights.keys())
        values = [float(weights[i]) for i in indices]
        items.append(SparseResponseItem(indices=indices, values=values))
    return SparseResponse(vectors=items)


@app.post("/rerank", response_model=RerankResponse)
async def rerank(request: RerankRequest):
    try:
        _embedder, _reranker = _ensure_loaded()
    except RuntimeError:
        return JSONResponse({"detail": "Model is still loading"}, status_code=503)

    pairs = [[request.query, p] for p in request.passages]
    raw_scores = await asyncio.to_thread(_reranker.compute_score, pairs)
    if not isinstance(raw_scores, list):
        raw_scores = [float(raw_scores)]
    scores = _normalize_scores([float(s) for s in raw_scores])
    return RerankResponse(scores=scores)


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=8000)
