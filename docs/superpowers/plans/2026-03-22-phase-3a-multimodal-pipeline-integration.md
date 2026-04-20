# Phase 3A: Multimodal Pipeline Integration — Implementation Plan

**Spec:** `docs/superpowers/specs/2026-03-22-multimodal-stt-tts-vlm-data-pipeline-design.md` §7
**Depends on:** Phase 2A (ASR), Phase 2B (TTS), Phase 3B (VLM)
**Expected Duration:** 10-12 hours
**Deliverable:** Extended LangGraph pipeline with multimodal input/output + multi-collection retrieval

---

## Architecture

```
                    START
                      │
                      ▼
              ┌───────────────┐
              │normalize_input│  NEW
              └───────┬───────┘
                      │
          ┌───────────┼───────────┐
          ▼           ▼           ▼
      "audio"      "text"     "image"
          │           │           │
          ▼           │           ▼
    ┌──────────┐      │     ┌──────────┐
    │ asr_node │  NEW │     │vlm_query │  NEW
    │ audio→txt│      │     │ img→txt  │
    └────┬─────┘      │     └────┬─────┘
         │            │          │
         └────────────┼──────────┘
                      │
                      ▼  (query is now always text)
              ┌───────────────┐
              │  input_guard  │  EXISTING (no changes)
              └───────┬───────┘
                      │
              (existing pipeline: embed → retrieve → rerank
               → web_fallback → combine → generate
               → output_guard → citations)
                      │
                      ▼
              ┌───────────────┐
              │ format_output │  NEW
              │ if audio:     │
              │   call TTS    │
              └───────┬───────┘
                      │
                      ▼
                     END
```

## Multi-Collection Retrieval Update

```
Query → Embed (1024d)
  │
  ├──► econ_vn_news:   top-10 (dense+sparse RRF)
  ├──► econ_knowledge:  top-10 (dense+sparse RRF)
  │
  ▼
Merge 20 candidates → Reranker → top-5
```

## Tasks

### Task 1: Extend RAGState (1h)

**Modify:** `api/orchestrator/pipeline/rag.py`
- Add new fields to `RAGState` TypedDict:
  - `input_modality: str` ("text" | "audio" | "image")
  - `audio_file: bytes | None`
  - `image_data: str | None`
  - `response_format: str` ("text" | "audio" | "text+audio")
  - `audio_response: bytes | None`

### Task 2: normalize_input Node (2h)

**Modify:** `api/orchestrator/pipeline/rag.py`
- New node `normalize_input`: detect modality from state
- Route function `route_by_modality` → "audio" | "image" | "text"

### Task 3: asr_node (1.5h)

**Modify:** `api/orchestrator/pipeline/rag.py`
- New node `asr_node`: call `services.asr.transcribe(state["audio_file"])`
- Sets `state["query"]` to transcribed text
- Error handling: set `state["error"]` on failure

### Task 4: vlm_query_node (1.5h)

**Modify:** `api/orchestrator/pipeline/rag.py`
- New node `vlm_query_node`: call `services.vlm.analyze(state["image_data"])`
- Sets `state["query"]` to VLM description
- Error handling: set `state["error"]` on failure

### Task 5: format_output Node (2h)

**Modify:** `api/orchestrator/pipeline/rag.py`
- New node `format_output`: after citations
- If `response_format` includes "audio": call `services.tts.synthesize(state["answer"])`
- Set `state["audio_response"]` to WAV bytes
- Fallback: if TTS fails, leave `audio_response` as None

### Task 6: Multi-Collection Retrieval (1.5h)

**Modify:** `api/orchestrator/services/retriever.py`
- `RetrieverClient.hybrid_search()` accepts `collections: list[str]`
- Searches each collection in parallel (asyncio.gather)
- Merges results before returning
- Configurable via `RAG__COLLECTIONS` and `RAG__PER_COLLECTION_TOP_K`

### Task 7: Chat Endpoint Updates (1.5h)

**Modify:** `api/orchestrator/routers/chat.py`
- Accept multipart form data (audio file upload)
- Detect image in OpenAI-format content array (base64 `image_url`)
- Accept `response_format` field in request
- Include `audio` field in response (base64-encoded WAV) when requested

**Modify:** `api/orchestrator/models/schemas.py`
- Extend `ChatCompletionRequest` with `response_format: str = "text"`
- Extend `ChatCompletionResponse` with optional `audio: str | None`

### Task 8: Tests (2h)

**Create:** `tests/orchestrator/test_multimodal_pipeline.py`
- `test_audio_input_flows_through_asr`
- `test_image_input_flows_through_vlm`
- `test_text_input_skips_normalization`
- `test_format_output_calls_tts`
- `test_format_output_text_only_fallback`
- `test_multi_collection_retrieval`

### Verification

```bash
pytest tests/orchestrator/test_multimodal_pipeline.py -v

# Integration (requires all services)
docker compose --profile full up -d
# Audio query
curl -X POST http://localhost:8000/v1/chat/completions \
  -F "audio=@test_query.wav" \
  -F 'messages=[{"role":"user","content":"audio_query"}]' \
  -F 'response_format=text+audio'
```

---

## Summary

| Task | Description | Effort |
|------|-------------|--------|
| 1 | Extend RAGState | 1h |
| 2 | normalize_input Node | 2h |
| 3 | asr_node | 1.5h |
| 4 | vlm_query_node | 1.5h |
| 5 | format_output Node | 2h |
| 6 | Multi-Collection Retrieval | 1.5h |
| 7 | Chat Endpoint Updates | 1.5h |
| 8 | Tests | 2h |
| **Total** | **8 tasks** | **10-12h** |
