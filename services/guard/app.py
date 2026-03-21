# services/guard/app.py
import asyncio
import os
import re
from contextlib import asynccontextmanager
from typing import Literal

import torch
from fastapi import FastAPI
from fastapi.responses import JSONResponse
from pydantic import BaseModel, model_validator
from transformers import AutoModelForCausalLM, AutoTokenizer


MODEL_NAME = os.getenv("GUARD_MODEL", "Qwen/Qwen3Guard-Gen-0.6B")
model = None
tokenizer = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    global model, tokenizer
    tokenizer = await asyncio.to_thread(AutoTokenizer.from_pretrained, MODEL_NAME)
    model = await asyncio.to_thread(
        AutoModelForCausalLM.from_pretrained, MODEL_NAME,
        torch_dtype=torch.float16, device_map="auto"
    )
    model.eval()
    yield
    model = None
    tokenizer = None


app = FastAPI(title="Guard Service", lifespan=lifespan)


@app.get("/health")
async def health():
    if model is None:
        return JSONResponse({"status": "loading"}, status_code=503)
    return JSONResponse({"status": "ok"})


class ClassifyRequest(BaseModel):
    text: str
    role: Literal["input", "output"]
    prompt: str | None = None  # Required when role == "output"

    @model_validator(mode="after")
    def prompt_required_for_output(self) -> "ClassifyRequest":
        if self.role == "output" and not self.prompt:
            raise ValueError("prompt is required when role is 'output'")
        return self


class ClassifyResponse(BaseModel):
    label: str  # "safe" or "unsafe"


def parse_safety_label(output: str) -> str:
    """Parse 'Safety: Safe/Unsafe/Controversial' from model output."""
    match = re.search(r"Safety:\s*(Safe|Unsafe|Controversial)", output, re.IGNORECASE)
    if match:
        label = match.group(1).lower()
        return "safe" if label == "safe" else "unsafe"
    return "unsafe"  # Fail closed: unparseable → unsafe


def _run_classify(text: str, role: str, prompt: str | None) -> str:
    """Blocking classify logic — run in thread via asyncio.to_thread."""
    if role == "input":
        messages = [{"role": "user", "content": text}]
    else:
        messages = [
            {"role": "user", "content": prompt or ""},
            {"role": "assistant", "content": text},
        ]

    formatted = tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )
    inputs = tokenizer(formatted, return_tensors="pt").to(model.device)
    with torch.no_grad():
        output_ids = model.generate(**inputs, max_new_tokens=128)
    new_tokens = output_ids[0][inputs["input_ids"].shape[-1]:]
    output_text = tokenizer.decode(new_tokens, skip_special_tokens=True)
    return parse_safety_label(output_text)


@app.post("/classify", response_model=ClassifyResponse)
async def classify(request: ClassifyRequest):
    if model is None or tokenizer is None:
        return JSONResponse({"detail": "Model is still loading"}, status_code=503)
    label = await asyncio.to_thread(_run_classify, request.text, request.role, request.prompt)
    return ClassifyResponse(label=label)
