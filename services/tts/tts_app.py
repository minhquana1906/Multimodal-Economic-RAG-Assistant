"""TTS Service — VieNeu-TTS with on-demand GPU loading.

Endpoints:
    GET  /health      — Readiness check (idle/loading/ok)
    POST /synthesize  — Text → full WAV audio response
    POST /stream      — Text → chunked SSE audio stream
    POST /unload      — Explicit model unload to free VRAM
"""

import asyncio
import base64
import gc
import io
import json
import os
import sys
import time
from contextlib import asynccontextmanager

import numpy as np
import soundfile as sf
import torch
from fastapi import FastAPI
from fastapi.responses import JSONResponse, Response, StreamingResponse
from loguru import logger
from pydantic import BaseModel

from text_preprocessor import preprocess

logger.remove()
logger.add(sys.stderr, format="{time:HH:mm:ss} | {level} | {message}", level="INFO")

# ── Configuration ──────────────────────────────────────────────────────
TTS_MODEL = os.getenv("TTS_MODEL", "pnnbao-ump/VieNeu-TTS")
IDLE_TIMEOUT_S = int(os.getenv("TTS_IDLE_TIMEOUT", "300"))
SAMPLE_RATE = int(os.getenv("TTS_SAMPLE_RATE", "24000"))
DEFAULT_SPEED = float(os.getenv("TTS_SPEED", "1.0"))


# ── On-Demand Model Manager ───────────────────────────────────────────
class OnDemandModel:
    """Manages TTS model lifecycle: load on first request, unload after idle timeout."""

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
                    logger.info("Loading TTS model: {} ...", TTS_MODEL)
                    t0 = time.monotonic()
                    self.model = await asyncio.to_thread(self._load_model)
                    elapsed = time.monotonic() - t0
                    logger.info("TTS model loaded in {:.1f}s", elapsed)
                finally:
                    self._loading = False
            self._reset_idle_timer()
            return self.model

    def _load_model(self):
        """Blocking model load — run via asyncio.to_thread."""
        from vieneu import Vieneu

        return Vieneu(
            backbone_repo=TTS_MODEL,
            backbone_device="cuda:0",
            codec_device="cuda:0",
        )

    async def unload(self):
        """Unload model and free VRAM."""
        async with self._load_lock:
            await self._do_unload()

    async def _do_unload(self):
        """Internal unload (caller must hold _load_lock)."""
        if self.model is not None:
            logger.info("Unloading TTS model to free VRAM...")
            try:
                self.model.close()
            except Exception as e:
                logger.warning("Error during model close: {}", e)
            del self.model
            self.model = None
            torch.cuda.empty_cache()
            gc.collect()
            logger.info("TTS model unloaded, VRAM freed")
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
        "TTS service started (model={}, idle_timeout={}s, sample_rate={})",
        TTS_MODEL,
        IDLE_TIMEOUT_S,
        SAMPLE_RATE,
    )
    yield
    await on_demand.unload()


app = FastAPI(title="TTS Service", lifespan=lifespan)


# ── Request / Response Models ─────────────────────────────────────────
class SynthesizeRequest(BaseModel):
    text: str
    speed: float = DEFAULT_SPEED
    sample_rate: int = SAMPLE_RATE


# ── Helpers ───────────────────────────────────────────────────────────
def _audio_to_wav_bytes(audio: np.ndarray, sample_rate: int = SAMPLE_RATE) -> bytes:
    """Convert a float32 numpy waveform to WAV bytes in memory."""
    buf = io.BytesIO()
    sf.write(buf, audio, sample_rate, format="WAV")
    buf.seek(0)
    return buf.read()


def _resample_if_needed(audio: np.ndarray, orig_sr: int, target_sr: int) -> np.ndarray:
    """Resample audio if the requested sample rate differs from the model output."""
    if orig_sr == target_sr:
        return audio
    try:
        import torchaudio

        waveform = torch.from_numpy(audio).unsqueeze(0)
        resampler = torchaudio.transforms.Resample(orig_freq=orig_sr, new_freq=target_sr)
        resampled = resampler(waveform)
        return resampled.squeeze(0).numpy()
    except ImportError:
        logger.warning("torchaudio not available for resampling, returning native sample rate")
        return audio


# ── Health Endpoint ────────────────────────────────────────────────────
@app.get("/health")
async def health():
    if on_demand.is_loaded:
        return JSONResponse({"status": "ok", "model_loaded": True})
    if on_demand.is_loading:
        return JSONResponse({"status": "loading", "model_loaded": False})
    return JSONResponse({"status": "idle", "model_loaded": False})


# ── Synthesize Endpoint ───────────────────────────────────────────────
@app.post("/synthesize")
async def synthesize(req: SynthesizeRequest):
    """Synthesize full WAV audio from Vietnamese text."""
    if not req.text.strip():
        return JSONResponse({"detail": "Text must not be empty"}, status_code=400)

    # Preprocess text into sentences
    sentences = preprocess(req.text)
    if not sentences:
        return JSONResponse({"detail": "No speakable text after preprocessing"}, status_code=400)

    logger.info(
        "Synthesizing {} sentence(s), speed={}, sample_rate={}",
        len(sentences), req.speed, req.sample_rate,
    )

    # Get model (loads on-demand if needed)
    tts = await on_demand.get_model()

    # Synthesize each sentence and concatenate
    # Note: VieNeu-TTS does not expose a direct "speed" parameter.
    # We pass speed as temperature (higher = more variation / faster pacing).
    t0 = time.monotonic()
    audio_chunks: list[np.ndarray] = []

    for i, sentence in enumerate(sentences):
        try:
            chunk = await asyncio.to_thread(
                tts.infer,
                text=sentence,
                temperature=req.speed,
            )
            audio_chunks.append(chunk)
        except Exception as e:
            logger.error("TTS inference failed on sentence {}: {}", i, e)
            return JSONResponse(
                {"detail": f"Synthesis failed on sentence {i}: {e}"},
                status_code=500,
            )

    # Concatenate all chunks
    if not audio_chunks:
        return JSONResponse({"detail": "No audio generated"}, status_code=500)

    full_audio = np.concatenate(audio_chunks)

    # Resample if client requested a different sample rate
    output_sr = req.sample_rate
    full_audio = _resample_if_needed(full_audio, SAMPLE_RATE, output_sr)

    elapsed = time.monotonic() - t0
    duration_s = len(full_audio) / output_sr

    logger.info(
        "TTS: {} sentences, duration={:.1f}s, latency={:.1f}s",
        len(sentences),
        duration_s,
        elapsed,
    )

    # Convert to WAV
    wav_bytes = await asyncio.to_thread(_audio_to_wav_bytes, full_audio, output_sr)

    return Response(
        content=wav_bytes,
        media_type="audio/wav",
        headers={"X-Duration-Seconds": f"{duration_s:.2f}"},
    )


# ── Stream Endpoint (SSE) ─────────────────────────────────────────────
@app.post("/stream")
async def stream(req: SynthesizeRequest):
    """Stream audio chunks as SSE events, one per sentence."""
    if not req.text.strip():
        return JSONResponse({"detail": "Text must not be empty"}, status_code=400)

    sentences = preprocess(req.text)
    if not sentences:
        return JSONResponse({"detail": "No speakable text after preprocessing"}, status_code=400)

    logger.info(
        "Streaming {} sentence(s), speed={}, sample_rate={}",
        len(sentences), req.speed, req.sample_rate,
    )

    output_sr = req.sample_rate

    async def event_generator():
        tts = await on_demand.get_model()

        for i, sentence in enumerate(sentences):
            try:
                chunk = await asyncio.to_thread(
                    tts.infer,
                    text=sentence,
                    temperature=req.speed,
                )
                chunk = _resample_if_needed(chunk, SAMPLE_RATE, output_sr)
                wav_bytes = _audio_to_wav_bytes(chunk, output_sr)
                b64_audio = base64.b64encode(wav_bytes).decode("ascii")

                event_data = json.dumps(
                    {"audio": b64_audio, "sentence": sentence, "index": i},
                    ensure_ascii=False,
                )
                yield f"data: {event_data}\n\n"
            except Exception as e:
                logger.error("TTS stream failed on sentence {}: {}", i, e)
                error_data = json.dumps({"error": str(e), "index": i})
                yield f"data: {error_data}\n\n"
                break

        # Final done event
        yield f"data: {json.dumps({'done': True})}\n\n"

    return StreamingResponse(
        event_generator(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            "X-Accel-Buffering": "no",
        },
    )


# ── Unload Endpoint ────────────────────────────────────────────────────
@app.post("/unload")
async def unload():
    """Explicitly unload model to free VRAM (used by orchestrator before ASR)."""
    if not on_demand.is_loaded:
        return JSONResponse({"status": "already_idle"})
    await on_demand.unload()
    return JSONResponse({"status": "unloaded"})
