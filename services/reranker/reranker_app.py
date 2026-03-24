import asyncio
import os
import sys
from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field
import torch
from loguru import logger
from transformers import AutoModelForCausalLM, AutoTokenizer

logger.remove()
logger.add(sys.stderr, format="{time:HH:mm:ss} | {level} | {message}", level="INFO")

MODEL_NAME = os.getenv("RERANKER_MODEL", "Qwen/Qwen3-Reranker-0.6B")
model = None
tokenizer = None
token_true_id: int | None = None
token_false_id: int | None = None

# Chat template wrapping — required for correct logits
PREFIX = '<|im_start|>system\nJudge whether the Document meets the requirements based on the Query and the Instruct provided. Note that the answer can only be "yes" or "no".<|im_end|>\n<|im_start|>user\n'
SUFFIX = "<|im_end|>\n<|im_start|>assistant\n<think>\n\n</think>\n\n"
INSTRUCTION = "Cho một câu hỏi về kinh tế, tài chính, đánh giá mức độ liên quan của đoạn văn bản với câu hỏi"



@asynccontextmanager
async def lifespan(app: FastAPI):
    global model, tokenizer, token_true_id, token_false_id
    if AutoTokenizer is None or AutoModelForCausalLM is None:
        raise RuntimeError("transformers is required to load the reranker model")
    tokenizer = await asyncio.to_thread(
        AutoTokenizer.from_pretrained, MODEL_NAME, padding_side="left"
    )
    model = await asyncio.to_thread(
        AutoModelForCausalLM.from_pretrained,
        MODEL_NAME,
        torch_dtype="auto",
        device_map="auto",
    )
    model.eval()
    token_true_id = tokenizer.convert_tokens_to_ids("yes")
    token_false_id = tokenizer.convert_tokens_to_ids("no")
    logger.info(f"Reranker model loaded: {MODEL_NAME} | padding_side=left")
    yield
    model = None
    tokenizer = None


app = FastAPI(title="Reranker Service", lifespan=lifespan)


@app.get("/health")
async def health():
    if model is None:
        return JSONResponse({"status": "loading"}, status_code=503)
    return JSONResponse({"status": "ok"})


class RerankRequest(BaseModel):
    query: str
    passages: list[str] = Field(..., min_length=1)
    instruction: str | None = None  # falls back to module-level INSTRUCTION if None


class RerankResponse(BaseModel):
    scores: list[float]


def format_pair(query: str, document: str, instruction: str) -> str:
    body = f"<Instruct>: {instruction}\n<Query>: {query}\n<Document>: {document}"
    return PREFIX + body + SUFFIX


@app.post("/rerank", response_model=RerankResponse)
async def rerank(request: RerankRequest):
    if model is None or tokenizer is None:
        return JSONResponse({"detail": "Model is still loading"}, status_code=503)

    effective_instruction = request.instruction or INSTRUCTION

    def _score_passage(passage: str) -> float:
        formatted = format_pair(request.query, passage, effective_instruction)
        inputs = tokenizer(formatted, return_tensors="pt", padding=True).to(
            model.device
        )
        with torch.no_grad():
            logits = model(**inputs).logits[:, -1, :]
            true_score = logits[:, token_true_id].exp().item()
            false_score = logits[:, token_false_id].exp().item()
            if (true_score + false_score) == 0:
                return 0.5  # fallback when underflow occurs
            return true_score / (true_score + false_score)

    # Note: passages are scored sequentially. For high-throughput production,
    # consider batching all passages in a single forward pass.
    scores = []
    for passage in request.passages:
        score = await asyncio.to_thread(_score_passage, passage)
        scores.append(score)
    return RerankResponse(scores=scores)
