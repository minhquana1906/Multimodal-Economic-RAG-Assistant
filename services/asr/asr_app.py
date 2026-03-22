"""ASR Service — Qwen3-ASR-1.7B with on-demand GPU loading.

Endpoints:
    GET  /health      — Readiness check (idle/loading/ok)
    POST /transcribe  — Audio → text transcription
    POST /unload      — Explicit model unload to free VRAM
"""

import asyncio
import gc
import io
import os
import sys
import time
from contextlib import asynccontextmanager

import torch
import torchaudio
from fastapi import FastAPI, File, Form, UploadFile
from fastapi.responses import JSONResponse
from loguru import logger
from pydantic import BaseModel

logger.remove()
logger.add(sys.stderr, format="{time:HH:mm:ss} | {level} | {message}", level="INFO")

# ── Configuration ──────────────────────────────────────────────────────
MODEL_NAME = os.getenv("ASR_MODEL", "Qwen/Qwen3-ASR-1.7B")
MAX_DURATION_S = int(os.getenv("ASR_MAX_DURATION_S", "60"))
IDLE_TIMEOUT_S = int(os.getenv("ASR_IDLE_TIMEOUT", "300"))
TARGET_SAMPLE_RATE = 16_000


# ── On-Demand Model Manager ───────────────────────────────────────────
class OnDemandModel:
    """Manages model lifecycle: load on first request, unload after idle timeout."""

    def __init__(self):
        self.model = None
        self._load_lock = asyncio.Lock()
        self._idle_task: asyncio.Task | None = None
        self._loading: bool = False

    @property
    def is_loaded(self) -> bool:
        return self.model is not None

    @property
    def is_loading(self) -> bool:
        return self._loading

    async def get_model(self):
        """Return loaded model, loading it first if necessary."""
        async with self._load_lock:
            if self.model is None:
                self._loading = True
                try:
                    logger.info("Loading ASR model: {} ...", MODEL_NAME)
                    t0 = time.monotonic()
                    self.model = await asyncio.to_thread(self._load_model)
                    elapsed = time.monotonic() - t0
                    logger.info("ASR model loaded in {:.1f}s", elapsed)
                finally:
                    self._loading = False
            self._reset_idle_timer()
            return self.model

    def _load_model(self):
        """Blocking model load — run via asyncio.to_thread."""
        from qwen_asr import Qwen3ASRModel

        return Qwen3ASRModel.from_pretrained(
            MODEL_NAME,
            dtype=torch.bfloat16,
            device_map="cuda:0",
            max_new_tokens=256,
        )

    async def unload(self):
        """Unload model and free VRAM."""
        async with self._load_lock:
            await self._do_unload()

    async def _do_unload(self):
        """Internal unload (caller must hold _load_lock)."""
        if self.model is not None:
            logger.info("Unloading ASR model to free VRAM...")
            del self.model
            self.model = None
            torch.cuda.empty_cache()
            gc.collect()
            logger.info("ASR model unloaded, VRAM freed")
        self._cancel_idle_timer()

    def _reset_idle_timer(self):
        """Reset the idle timeout. Must be called while holding _load_lock."""
        self._cancel_idle_timer()
        self._idle_task = asyncio.create_task(self._idle_countdown())

    def _cancel_idle_timer(self):
        if self._idle_task is not None:
            self._idle_task.cancel()
            self._idle_task = None

    async def _idle_countdown(self):
        """Wait for idle timeout, then unload."""
        try:
            await asyncio.sleep(IDLE_TIMEOUT_S)
            logger.info("Idle timeout ({}s) reached, unloading model...", IDLE_TIMEOUT_S)
            async with self._load_lock:
                await self._do_unload()
        except asyncio.CancelledError:
            pass  # Timer was reset by a new request


# ── App Setup ──────────────────────────────────────────────────────────
on_demand = OnDemandModel()


@asynccontextmanager
async def lifespan(app: FastAPI):
    logger.info(
        "ASR service started (model={}, idle_timeout={}s, max_duration={}s)",
        MODEL_NAME,
        IDLE_TIMEOUT_S,
        MAX_DURATION_S,
    )
    yield
    await on_demand.unload()


app = FastAPI(title="ASR Service", lifespan=lifespan)


# ── Health Endpoint ────────────────────────────────────────────────────
@app.get("/health")
async def health():
    if on_demand.is_loaded:
        return JSONResponse({"status": "ok", "model_loaded": True})
    if on_demand.is_loading:
        return JSONResponse({"status": "loading", "model_loaded": False})
    return JSONResponse({"status": "idle", "model_loaded": False})
