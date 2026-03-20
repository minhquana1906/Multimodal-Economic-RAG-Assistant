from contextlib import asynccontextmanager
from fastapi import FastAPI
from fastapi.responses import JSONResponse
from sentence_transformers import SentenceTransformer
import os

MODEL_NAME = os.getenv("EMBEDDING_MODEL", "Qwen/Qwen3-Embedding-0.6B")
model: SentenceTransformer | None = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    global model
    model = SentenceTransformer(MODEL_NAME)
    yield
    model = None


app = FastAPI(title="Embedding Service", lifespan=lifespan)


@app.get("/health")
async def health():
    if model is None:
        return JSONResponse({"status": "loading"}, status_code=503)
    return {"status": "ok"}
