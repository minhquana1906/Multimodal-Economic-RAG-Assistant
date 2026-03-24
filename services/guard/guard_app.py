# services/guard/guard_app.py
import asyncio
import os
import re
import sys
import time
from contextlib import asynccontextmanager
from typing import Literal

import torch
from fastapi import FastAPI
from fastapi.responses import JSONResponse
from loguru import logger
from pydantic import BaseModel, model_validator
from transformers import AutoModelForCausalLM, AutoTokenizer

logger.remove()
logger.add(sys.stderr, format="{time:HH:mm:ss} | {level} | {message}", level="INFO")

MODEL_NAME = os.getenv("GUARD_MODEL", "Qwen/Qwen3Guard-Gen-0.6B")
MAX_NEW_TOKENS = int(os.getenv("GUARD_MAX_NEW_TOKENS", "64"))
SAFETY_LABELS = {"safe": "Safe", "unsafe": "Unsafe", "controversial": "Controversial"}
CATEGORY_NAMES = [
    "Non-violent Illegal Acts",
    "Sexual Content or Sexual Acts",
    "Suicide & Self-Harm",
    "Politically Sensitive Topics",
    "Copyright Violation",
    "Unethical Acts",
    "Jailbreak",
    "Violent",
    "PII",
    "None",
]
CATEGORY_PATTERN = re.compile(
    rf"(?:Category:\s*)?({'|'.join(re.escape(name) for name in CATEGORY_NAMES)})",
    re.IGNORECASE,
)
model = None
tokenizer = None


def _cuda_memory_value(name: str) -> int:
    cuda = getattr(torch, "cuda", None)
    if cuda is None:
        return 0
    is_available = getattr(cuda, "is_available", lambda: False)
    if not is_available():
        return 0
    getter = getattr(cuda, name, None)
    if getter is None:
        return 0
    try:
        return int(getter())
    except Exception:
        return 0


def _log_request_metrics(operation: str, started_at: float) -> None:
    latency_ms = round((time.perf_counter() - started_at) * 1000, 2)
    logger.info(
        "{} latency_ms={} memory_allocated={} memory_reserved={} max_memory_allocated={}",
        operation,
        latency_ms,
        _cuda_memory_value("memory_allocated"),
        _cuda_memory_value("memory_reserved"),
        _cuda_memory_value("max_memory_allocated"),
    )


def _cleanup_tensors(*tensors: object) -> None:
    for tensor in tensors:
        del tensor


@asynccontextmanager
async def lifespan(app: FastAPI):
    global model, tokenizer
    if AutoTokenizer is None or AutoModelForCausalLM is None:
        raise RuntimeError("transformers is required to load the guard model")
    tokenizer = await asyncio.to_thread(AutoTokenizer.from_pretrained, MODEL_NAME)
    model = await asyncio.to_thread(
        AutoModelForCausalLM.from_pretrained,
        MODEL_NAME,
        torch_dtype="auto",
        device_map="auto",
    )
    model.eval()
    logger.info(f"Guard model loaded: {MODEL_NAME}")
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
    prompt: str | None = None

    @model_validator(mode="after")
    def prompt_required_for_output(self) -> "ClassifyRequest":
        if self.role == "output" and not self.prompt:
            raise ValueError("prompt is required when role is 'output'")
        return self


class ClassifyResponse(BaseModel):
    label: str
    safe_label: str | None = None
    categories: list[str] = []
    refusal: str | None = None


def extract_label_categories_refusal(content: str) -> tuple[str | None, list[str], str | None]:
    safe_match = re.search(r"Safety:\s*(Safe|Unsafe|Controversial)", content, re.IGNORECASE)
    refusal_match = re.search(r"Refusal:\s*(Yes|No)", content, re.IGNORECASE)

    safe_label = None
    if safe_match:
        safe_label = SAFETY_LABELS[safe_match.group(1).lower()]

    categories: list[str] = []
    for match in CATEGORY_PATTERN.findall(content):
        normalized = next(
            (name for name in CATEGORY_NAMES if name.lower() == match.lower()),
            match,
        )
        if normalized == "None" or normalized in categories:
            continue
        categories.append(normalized)

    refusal = refusal_match.group(1).capitalize() if refusal_match else None
    return safe_label, categories, refusal


def parse_safety_label(output: str) -> str:
    """Parse 'Safety: Safe/Unsafe/Controversial' from model output."""
    safe_label, _, _ = extract_label_categories_refusal(output)
    if safe_label == "Safe":
        return "safe"
    return "unsafe"


def _build_classification_result(output_text: str) -> dict:
    safe_label, categories, refusal = extract_label_categories_refusal(output_text)
    return {
        "label": "safe" if safe_label == "Safe" else "unsafe",
        "safe_label": safe_label,
        "categories": categories,
        "refusal": refusal,
    }


def _run_classify(text: str, role: str, prompt: str | None) -> dict:
    """Blocking classify logic — run in thread via asyncio.to_thread."""
    started_at = time.perf_counter()
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
    inputs = None
    output_ids = None
    new_tokens = None

    try:
        inputs = tokenizer(formatted, return_tensors="pt").to(model.device)
        with torch.inference_mode():
            output_ids = model.generate(
                **inputs,
                max_new_tokens=MAX_NEW_TOKENS,
                use_cache=False,
            )
        new_tokens = output_ids[0][inputs["input_ids"].shape[-1] :]
        output_text = tokenizer.decode(new_tokens, skip_special_tokens=True)
        return _build_classification_result(output_text)
    finally:
        _cleanup_tensors(inputs, output_ids, new_tokens)
        inputs = None
        output_ids = None
        new_tokens = None
        _log_request_metrics("guard.classify", started_at)


@app.post("/classify", response_model=ClassifyResponse)
async def classify(request: ClassifyRequest):
    if model is None or tokenizer is None:
        return JSONResponse({"detail": "Model is still loading"}, status_code=503)
    result = await asyncio.to_thread(
        _run_classify, request.text, request.role, request.prompt
    )
    return ClassifyResponse(**result)
