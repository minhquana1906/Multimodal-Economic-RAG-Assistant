# Phase 2B: TTS Service — Implementation Plan

**Spec:** `docs/superpowers/specs/2026-03-22-multimodal-stt-tts-vlm-data-pipeline-design.md` §4
**Depends on:** Phase 1 (complete)
**Expected Duration:** 8-10 hours
**Deliverable:** Working TTS microservice with Vietnamese text preprocessing

---

## Architecture

```
┌─────────────────────────────────────────────────────┐
│              TTS Service (port 8006)                  │
│                                                       │
│   Vietnamese Text                                     │
│         │                                             │
│         ▼                                             │
│   ┌──────────────┐    ┌──────────────────────┐       │
│   │ Text Preproc  │───►│ VieNeu-1000h Model   │       │
│   │ • Normalize   │    │ (VITS2/FastSpeech2)  │       │
│   │   numbers     │    │ GPU, on-demand       │       │
│   │ • Expand      │    └──────┬───────────────┘       │
│   │   abbrevs     │           │                       │
│   │ • Split sents │           ▼                       │
│   └──────────────┘    WAV Audio Buffer                │
│                              │                        │
│                    ┌─────────┼─────────┐              │
│                    ▼                   ▼              │
│              POST /synthesize    POST /stream         │
│              (full WAV)          (chunked SSE)        │
└─────────────────────────────────────────────────────┘
```

## Vietnamese Text Preprocessing Flow

```
Input: "GDP tăng 6.5% trong Q3/2023, theo NHNN."
  │
  ▼
Step 1: Number normalization
"GDP tăng sáu phẩy năm phần trăm trong quý ba năm hai nghìn không trăm hai mươi ba"
  │
  ▼
Step 2: Abbreviation expansion
"Tổng sản phẩm quốc nội tăng sáu phẩy năm phần trăm ... Ngân hàng Nhà nước"
  │
  ▼
Step 3: Clean markdown/citations
(remove [1], **bold**, URLs, etc.)
  │
  ▼
Step 4: Sentence split → per-sentence synthesis → concatenate WAV
```

## Tasks

### Task 1: Research VieNeu-1000h (2h)

- Determine exact model loading API (VITS2 or FastSpeech2 variant)
- Test inference locally: text in → WAV out
- Measure VRAM usage and latency
- Identify required preprocessing steps

### Task 2: TTS Service Core (3h)

**Create:** `services/tts/tts_app.py`
- OnDemandModel pattern (same as ASR)
- POST /synthesize → full WAV response
- POST /stream → chunked audio SSE
- GET /health

**Create:** `services/tts/text_preprocessor.py`
- `normalize_numbers(text) -> str` — Vietnamese number reading
- `expand_abbreviations(text, abbrev_dict) -> str`
- `clean_for_speech(text) -> str` — strip markdown, citations
- `split_sentences(text) -> list[str]`

**Create:** `services/tts/abbreviations.py`
- Dictionary of Vietnamese economic abbreviations
- Examples: GDP, GNP, CPI, FDI, NHNN, TTCK, BCTC, etc.

### Task 3: Dockerfile & Dependencies (1h)

**Create:** `services/tts/Dockerfile`
**Create:** `services/tts/requirements.txt`

### Task 4: TTS Client (1h)

**Create:** `api/orchestrator/services/tts.py`
- `TTSClient(url, timeout)`
- `async def synthesize(text, speed) -> bytes | None`
- Returns None on error (graceful fallback to text-only)

### Task 5: Docker Compose (0.5h)

**Update:** `docker-compose.yml`
- Add `tts` service under `profiles: [audio]`
- Same GPU pattern as ASR

### Task 6: Tests (1.5h)

**Create:** `tests/services/test_tts.py`
- `test_health_endpoint`
- `test_synthesize_returns_wav`
- `test_text_preprocessor_numbers`
- `test_text_preprocessor_abbreviations`

**Create:** `tests/orchestrator/test_tts_client.py`
- `test_tts_client_synthesize`
- `test_tts_client_returns_none_on_error`

### Verification

```bash
pytest tests/services/test_tts.py tests/orchestrator/test_tts_client.py -v

docker compose --profile audio up -d tts
curl http://localhost:8006/health
curl -X POST http://localhost:8006/synthesize \
  -H "Content-Type: application/json" \
  -d '{"text": "GDP Việt Nam tăng trưởng 6.5% trong năm 2023.", "speed": 1.0}' \
  --output test_output.wav
```

---

## Summary

| Task | Description | Effort |
|------|-------------|--------|
| 1 | Research VieNeu-1000h | 2h |
| 2 | TTS Service Core | 3h |
| 3 | Dockerfile & Dependencies | 1h |
| 4 | TTS Client | 1h |
| 5 | Docker Compose | 0.5h |
| 6 | Tests | 1.5h |
| **Total** | **6 tasks** | **8-10h** |
