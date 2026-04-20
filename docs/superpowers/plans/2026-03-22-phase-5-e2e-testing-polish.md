# Phase 5: E2E Testing & Polish — Implementation Plan

**Spec:** `docs/superpowers/specs/2026-03-22-multimodal-stt-tts-vlm-data-pipeline-design.md`
**Depends on:** ALL phases (2A, 2B, 2C, 3A, 3B, 4)
**Expected Duration:** 5-8 hours
**Deliverable:** Production-ready multimodal system with E2E tests

---

## Architecture Verification

```
┌───────────────────────────────────────────────────────────────────┐
│                     E2E TEST SCENARIOS                             │
│                                                                    │
│  Test 1: Voice Query → Text+Audio Response                        │
│  ┌────────┐    ┌─────┐    ┌───────────────┐    ┌─────┐          │
│  │Audio.wav├───►│ ASR ├───►│ RAG Pipeline  ├───►│ TTS ├──►WAV   │
│  └────────┘    └─────┘    └───────────────┘    └─────┘          │
│                                                                    │
│  Test 2: Image Query → Text Response                              │
│  ┌────────┐    ┌─────┐    ┌───────────────┐                      │
│  │Chart.png├───►│ VLM ├───►│ RAG Pipeline  ├──►Text+Citations    │
│  └────────┘    └─────┘    └───────────────┘                      │
│                                                                    │
│  Test 3: Multi-Collection Retrieval                               │
│  ┌──────┐    ┌────────────────┐    ┌────────────────┐            │
│  │Query ├───►│ econ_vn_news   ├───►│                │            │
│  │      │    │ econ_knowledge ├───►│ Rerank → LLM   ├──►Answer  │
│  └──────┘    └────────────────┘    └────────────────┘            │
│                                                                    │
│  Test 4: Data Pipeline E2E                                        │
│  ┌──────┐    ┌───────┐    ┌──────┐    ┌───────┐    ┌──────┐    │
│  │Crawl ├───►│ Parse ├───►│ VLM  ├───►│ Chunk ├───►│Qdrant│    │
│  └──────┘    └───────┘    └──────┘    └───────┘    └──────┘    │
│                                                                    │
│  Test 5: VRAM Budget Verification                                 │
│  ┌────────────────────────────────────────────┐                  │
│  │ nvidia-smi monitoring during:              │                  │
│  │ - All 3 base services loaded (3.6GB)       │                  │
│  │ - ASR cold start + inference (+3.4GB)      │                  │
│  │ - ASR idle unload (-3.4GB)                 │                  │
│  │ - TTS cold start + inference (+2-3GB)      │                  │
│  │ - ASR+TTS simultaneous (peak ~10GB)        │                  │
│  └────────────────────────────────────────────┘                  │
└───────────────────────────────────────────────────────────────────┘
```

## Tasks

### Task 1: E2E Voice Query Test (2h)

**Create:** `tests/e2e/test_voice_query.py`
- Record/generate a test WAV file with Vietnamese speech
- Send to `/v1/chat/completions` as multipart audio
- Verify: ASR transcription → RAG retrieval → LLM answer → TTS audio
- Assert: response contains both text and audio (base64 WAV)
- Assert: transcription is reasonable Vietnamese text
- Assert: audio duration > 0

### Task 2: E2E Image Query Test (1h)

**Create:** `tests/e2e/test_image_query.py`
- Use a sample economic chart image (PNG)
- Send as base64 in OpenAI-format content array
- Verify: VLM description → RAG retrieval → LLM answer
- Assert: answer references chart content
- Assert: citations present

### Task 3: E2E Multi-Collection Test (1h)

**Create:** `tests/e2e/test_multi_collection.py`
- Seed both `econ_vn_news` and `econ_knowledge` with test data
- Query that should match documents from both collections
- Verify: results from both collections appear in retrieved_docs
- Verify: reranker correctly ranks across collections

### Task 4: VRAM Profiling (2h)

**Create:** `scripts/vram_profile.py`
- Monitor `nvidia-smi` GPU memory during:
  - Base services loaded (embedding + reranker + guard)
  - ASR cold start → inference → idle timeout → unload
  - TTS cold start → inference → idle timeout → unload
  - Simultaneous ASR + TTS (peak stress test)
- Log VRAM usage at each stage
- Assert: never exceeds 11.5 GB (leave 0.5GB headroom)
- Output: VRAM usage report

### Task 5: Docker Compose Profiles (1h)

**Verify:** `docker-compose.yml`
- `docker compose up -d` — default (text-only)
- `docker compose --profile audio up -d` — adds ASR + TTS
- `docker compose --profile full up -d` — everything
- `docker compose --profile crawl up -d` — data pipeline
- All health checks pass
- No orphan containers

### Task 6: Config & Env Updates (0.5h)

**Update:** `api/orchestrator/config.py`
- Add ASR, TTS, VLM config groups to Settings
- Add multi-collection config

**Update:** `.env.example`
- Add all new env vars with comments

### Task 7: Documentation (0.5h)

**Update:** `docs/superpowers/plans/README.md`
- Add new phase plans to the index
- Update phase dependency diagram
- Mark Phase 1 as complete, new phases as ready

### Verification

```bash
# Full E2E suite
docker compose --profile full up -d
pytest tests/e2e/ -v --timeout=120

# VRAM profile
python scripts/vram_profile.py

# Profile tests
docker compose down
docker compose up -d  # default only
docker compose --profile audio up -d  # + audio
docker compose --profile full up -d  # + everything
```

---

## Summary

| Task | Description | Effort |
|------|-------------|--------|
| 1 | E2E Voice Query Test | 2h |
| 2 | E2E Image Query Test | 1h |
| 3 | E2E Multi-Collection Test | 1h |
| 4 | VRAM Profiling | 2h |
| 5 | Docker Compose Profiles | 1h |
| 6 | Config & Env Updates | 0.5h |
| 7 | Documentation | 0.5h |
| **Total** | **7 tasks** | **5-8h** |
