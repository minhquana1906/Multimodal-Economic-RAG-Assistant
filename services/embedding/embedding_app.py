import asyncio
import os
import sys
from contextlib import asynccontextmanager

import torch
from fastapi import FastAPI
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field
from sentence_transformers import SentenceTransformer
from loguru import logger

logger.remove()
logger.add(sys.stderr, format="{time:HH:mm:ss} | {level} | {message}", level="INFO")

MODEL_NAME = os.getenv("EMBEDDING_MODEL", "Qwen/Qwen3-Embedding-0.6B")
MAX_SEQ_LENGTH = int(os.getenv("MAX_SEQ_LENGTH", "1024"))
ENCODE_BATCH_SIZE = int(os.getenv("ENCODE_BATCH_SIZE", "128"))

model: SentenceTransformer | None = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    global model
    model = await asyncio.to_thread(SentenceTransformer, MODEL_NAME)
    model.max_seq_length = MAX_SEQ_LENGTH
    logger.info(
        f"Model loaded: {MODEL_NAME} | "
        f"max_seq_length={model.max_seq_length} | "
        f"encode_batch_size={ENCODE_BATCH_SIZE}"
    )
    yield
    model = None


app = FastAPI(title="Embedding Service", lifespan=lifespan)


@app.get("/health")
async def health():
    if model is None:
        return JSONResponse({"status": "loading"}, status_code=503)
    return JSONResponse(
        {
            "status": "ok",
            "model": MODEL_NAME,
            "max_seq_length": model.max_seq_length,
            "encode_batch_size": ENCODE_BATCH_SIZE,
            "device": str(model.device),
            "dtype": str(model.dtype),
            "num_embeddings": model.get_sentence_embedding_dimension(),
        }
    )


class EmbedRequest(BaseModel):
    texts: list[str] = Field(..., min_length=1)
    is_query: bool = False


class EmbedResponse(BaseModel):
    embeddings: list[list[float]]


def _encode(texts, is_query):
    """Encode texts with OOM-safe fallback and periodic cache cleanup."""
    kwargs = {"batch_size": ENCODE_BATCH_SIZE, "show_progress_bar": False}
    if is_query:
        kwargs["prompt_name"] = "query"
    try:
        result = model.encode(texts, **kwargs)
    except torch.cuda.OutOfMemoryError:
        # Clear fragmented CUDA cache and retry one-by-one
        logger.warning(
            f"OOM on batch of {len(texts)} texts, clearing cache and retrying one-by-one"
        )
        torch.cuda.empty_cache()
        results = []
        for text in texts:
            single = model.encode([text], **kwargs)
            results.append(single[0])
            torch.cuda.empty_cache()
        import numpy as np

        result = np.stack(results)
    finally:
        # Prevent fragmentation buildup across many requests
        torch.cuda.empty_cache()
    return result


@app.post("/embed", response_model=EmbedResponse)
async def embed(request: EmbedRequest):
    if model is None:
        return JSONResponse({"detail": "Model is still loading"}, status_code=503)
    vectors = await asyncio.to_thread(_encode, request.texts, request.is_query)
    return EmbedResponse(embeddings=vectors.tolist())
