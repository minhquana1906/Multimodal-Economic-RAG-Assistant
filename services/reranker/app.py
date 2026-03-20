import asyncio
import os
from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field
from transformers import AutoModelForCausalLM, AutoTokenizer
import torch


MODEL_NAME = os.getenv("RERANKER_MODEL", "Qwen/Qwen3-Reranker-0.6B")
model = None
tokenizer = None
token_true_id: int = 0
token_false_id: int = 0

# Chat template wrapping — required for correct logits
PREFIX = '<|im_start|>system\nJudge whether the Document meets the requirements based on the Query and the Instruct provided. Note that the answer can only be "yes" or "no".<|im_end|>\n<|im_start|>user\n'
SUFFIX = '<|im_end|>\n<|im_start|>assistant\n<think>\n\n</think>\n\n'
INSTRUCTION = "Cho một câu hỏi về kinh tế, tài chính, đánh giá mức độ liên quan của đoạn văn bản với câu hỏi"


@asynccontextmanager
async def lifespan(app: FastAPI):
    global model, tokenizer, token_true_id, token_false_id
    tokenizer = await asyncio.to_thread(
        AutoTokenizer.from_pretrained, MODEL_NAME, padding_side="left"
    )
    model = await asyncio.to_thread(
        AutoModelForCausalLM.from_pretrained, MODEL_NAME,
        torch_dtype=torch.float16, device_map="auto"
    )
    model.eval()
    token_true_id = tokenizer.convert_tokens_to_ids("yes")
    token_false_id = tokenizer.convert_tokens_to_ids("no")
    yield
    model = None
    tokenizer = None


app = FastAPI(title="Reranker Service", lifespan=lifespan)


@app.get("/health")
async def health():
    if model is None:
        return JSONResponse({"status": "loading"}, status_code=503)
    return JSONResponse({"status": "ok"})
