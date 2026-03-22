import asyncio
import os
import sys
from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field
from sentence_transformers import SentenceTransformer
from loguru import logger

logger.remove()
logger.add(sys.stderr, format="{time:HH:mm:ss} | {level} | {message}", level="INFO")

MODEL_NAME = os.getenv("EMBEDDING_MODEL", "Qwen/Qwen3-Embedding-0.6B")
model: SentenceTransformer | None = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    global model
    model = await asyncio.to_thread(SentenceTransformer, MODEL_NAME)
    yield
    model = None


app = FastAPI(title="Embedding Service", lifespan=lifespan)


@app.get("/health")
async def health():
    if model is None:
        return JSONResponse({"status": "loading"}, status_code=503)
    return JSONResponse({"status": "ok"})


class EmbedRequest(BaseModel):
    texts: list[str] = Field(..., min_length=1)
    is_query: bool = False


class EmbedResponse(BaseModel):
    embeddings: list[list[float]]


def _encode(texts, is_query):
    if is_query:
        return model.encode(texts, prompt_name="query")
    return model.encode(texts)


@app.post("/embed", response_model=EmbedResponse)
async def embed(request: EmbedRequest):
    if model is None:
        return JSONResponse({"detail": "Model is still loading"}, status_code=503)
    vectors = await asyncio.to_thread(_encode, request.texts, request.is_query)
    return EmbedResponse(embeddings=vectors.tolist())
