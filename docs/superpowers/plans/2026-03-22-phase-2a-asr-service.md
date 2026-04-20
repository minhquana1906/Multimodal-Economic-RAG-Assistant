# Phase 2A: ASR Service — Implementation Plan

**Spec:** `docs/superpowers/specs/2026-03-22-multimodal-stt-tts-vlm-data-pipeline-design.md` §3
**Depends on:** Phase 1 (complete)
**Expected Duration:** 8-10 hours
**Deliverable:** Working ASR microservice with on-demand GPU loading

---

## Architecture

```
┌─────────────────────────────────────────────────────┐
│              ASR Service (port 8005)                  │
│                                                       │
│   Audio Input (WAV/MP3/WebM/OGG/FLAC)               │
│         │                                             │
│         ▼                                             │
│   ┌─────────────┐     ┌──────────────────┐           │
│   │ Audio Decode │────►│ Qwen3-ASR-1.7B   │           │
│   │ (torchaudio  │     │ (qwen-asr lib)   │           │
│   │  / ffmpeg)   │     │ GPU, on-demand   │           │
│   └─────────────┘     └──────┬───────────┘           │
│                              │                        │
│                              ▼                        │
│                     Vietnamese Text                   │
│                                                       │
│   POST /transcribe  ← audio → text                  │
│   GET  /health      ← readiness check               │
└─────────────────────────────────────────────────────┘
```

## On-Demand Model Loading Flow

```
Request ──► Model loaded? ──► YES ──► Inference ──► Response
                │                                      │
                NO                              Reset idle timer
                │
                ▼
         Load model (~5-8s)
                │
                ▼
         Inference ──► Response
                │
         Start idle timer (300s)
                │
         Timer expires ──► Unload + cuda.empty_cache()
```

## Tasks

### Task 1: ASR Service Core (3h)

**Create:** `services/asr/asr_app.py`

```python
# Key components:
# - FastAPI app with lifespan
# - OnDemandModel class (load/unload/idle timeout)
# - POST /transcribe endpoint (multipart form)
# - GET /health endpoint
# - Audio format validation + conversion to 16kHz mono WAV
# - Max duration check (ASR_MAX_DURATION_S, default 60)
```

**Implementation notes:**
- Use `qwen-asr` package for Qwen3-ASR-1.7B inference
- `torchaudio.load()` for decoding + resampling
- `threading.Timer` for idle timeout
- `torch.cuda.empty_cache()` after model unload
- Loguru logging with `ASR` domain level

### Task 2: Dockerfile & Dependencies (1h)

**Create:** `services/asr/Dockerfile`
- Base: `pytorch/pytorch:2.1.1-cuda12.1-cudnn8-runtime`
- Install: `qwen-asr`, `torchaudio`, `fastapi`, `uvicorn`, `python-multipart`, `loguru`

**Create:** `services/asr/requirements.txt`

### Task 3: ASR Client (1.5h)

**Create:** `api/orchestrator/services/asr.py`
- `ASRClient(url, timeout)`
- `async def transcribe(audio_bytes, content_type, language) -> str`
- Multipart upload via `httpx`
- Error handling: `ASRError` on failure

### Task 4: Docker Compose (0.5h)

**Update:** `docker-compose.yml`
- Add `asr` service under `profiles: [audio]`
- GPU allocation, health check, env vars
- Depends on nothing (standalone)

### Task 5: Tests (2h)

**Create:** `tests/services/test_asr.py`
- `test_health_endpoint`
- `test_transcribe_returns_text`
- `test_transcribe_rejects_too_long_audio`
- `test_model_idle_unload`

**Create:** `tests/orchestrator/test_asr_client.py`
- `test_asr_client_transcribes`
- `test_asr_client_handles_errors`

### Verification

```bash
# Unit tests
pytest tests/services/test_asr.py tests/orchestrator/test_asr_client.py -v

# Integration (requires GPU)
docker compose --profile audio up -d asr
curl http://localhost:8005/health
curl -X POST http://localhost:8005/transcribe -F "file=@test_audio.wav" -F "language=vi"
```

---

## Summary

| Task | Description | Effort |
|------|-------------|--------|
| 1 | ASR Service Core | 3h |
| 2 | Dockerfile & Dependencies | 1h |
| 3 | ASR Client | 1.5h |
| 4 | Docker Compose | 0.5h |
| 5 | Tests | 2h |
| **Total** | **5 tasks** | **8-10h** |
