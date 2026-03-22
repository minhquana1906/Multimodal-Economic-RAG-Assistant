# Phase 3B: VLM Service — Implementation Plan

**Spec:** `docs/superpowers/specs/2026-03-22-multimodal-stt-tts-vlm-data-pipeline-design.md` §5
**Depends on:** Phase 1 (complete)
**Expected Duration:** 10-12 hours
**Deliverable:** Working VLM service (remote) + VLM enricher for data pipeline

---

## Architecture

```
┌─────────────────────────────────────────────────────────────┐
│         VLM Service (port 8007, remote vast.ai)              │
│                                                               │
│         Qwen3-VL-4B-Instruct (~8GB VRAM)                    │
│                                                               │
│   ┌──────────────────────────────────────────────────┐      │
│   │                                                    │      │
│   │  POST /analyze         ← general image analysis   │      │
│   │  POST /extract-table   ← table → markdown         │      │
│   │  POST /describe-chart  ← chart → text + insights  │      │
│   │  GET  /health                                      │      │
│   │                                                    │      │
│   └──────────────────────────────────────────────────┘      │
│                                                               │
│   Dual purpose:                                               │
│   1. Runtime: user uploads image → analyze → RAG query       │
│   2. Ingestion: PDF charts/tables → describe → Qdrant chunks │
└─────────────────────────────────────────────────────────────┘
```

## Dual Storage for Visual Elements

```
PDF/Web contains chart
        │
        ▼
  Extract image (PNG) ──► Save to data/images/{hash}.png
        │
        ▼
  VLM /describe-chart
        │
        ▼
  Qdrant chunk:
    text = VLM description (searchable via embedding)
    payload.image_path = "data/images/{hash}.png"
    payload.chunk_type = "chart"
    payload.structured_data = {trend, key_insights}
```

## Tasks

### Task 1: VLM Service Core (4h)

**Create:** `services/vlm/vlm_app.py`

Key implementation:
- Load `Qwen3-VL-4B-Instruct` via `transformers` (`Qwen2_5_VLForConditionalGeneration` or equivalent)
- Image preprocessing: decode base64, resize to max 1024x1024, normalize
- Three specialized endpoints with different system prompts:

```python
# /analyze — general purpose
ANALYZE_PROMPT = "Mô tả chi tiết nội dung hình ảnh này."

# /extract-table — table extraction
TABLE_PROMPT = """Trích xuất bảng này thành định dạng markdown table.
Giữ nguyên tất cả số liệu và headers. Nếu có đơn vị, ghi rõ đơn vị."""

# /describe-chart — chart analysis
CHART_PROMPT = """Phân tích biểu đồ kinh tế này. Mô tả:
1. Loại biểu đồ (cột, đường, tròn, etc.)
2. Tiêu đề và nhãn trục
3. Xu hướng chính
4. Các điểm dữ liệu quan trọng
5. Insight kinh tế rút ra được"""
```

- Loguru logging with `VLM` domain level

### Task 2: Dockerfile & Dependencies (1h)

**Create:** `services/vlm/Dockerfile`
- Base: `pytorch/pytorch:2.1.1-cuda12.1-cudnn8-runtime`
- Install: `transformers`, `accelerate`, `Pillow`, `fastapi`, `uvicorn`, `loguru`

**Create:** `services/vlm/requirements.txt`

### Task 3: VLM Client (1.5h)

**Create:** `api/orchestrator/services/vlm.py`
- `VLMClient(url, timeout, max_tokens)`
- `async def analyze(image_b64, prompt) -> dict`
- `async def extract_table(image_b64) -> dict`
- `async def describe_chart(image_b64) -> dict`
- Error handling: return empty dict on failure

### Task 4: VLM Enricher for Data Pipeline (2h)

**Create:** `scripts/pipeline/vlm_enricher.py`
- Batch process images through VLM service
- For each visual element in parsed documents:
  - Load image from `data/images/`
  - Classify: table or chart
  - Call appropriate VLM endpoint
  - Return enriched chunk data

```python
async def enrich_visual_elements(
    parsed_docs: list[ParsedDocument],
    vlm_client: VLMClient,
) -> list[EnrichedChunk]:
    """Process all visual elements through VLM, return text chunks."""
```

### Task 5: Docker/Remote Config (1h)

- VLM deployment script for vast.ai
- Update `.env.example` with `SERVICES__VLM_URL`
- Docker Compose entry (profile: vision, env vars only since remote)

### Task 6: Tests (2h)

**Create:** `tests/services/test_vlm.py`
- `test_health_endpoint`
- `test_analyze_returns_description`
- `test_extract_table_returns_markdown`
- `test_describe_chart_returns_insights`

**Create:** `tests/orchestrator/test_vlm_client.py`
- `test_vlm_client_analyze`
- `test_vlm_client_handles_errors`

**Create:** `tests/scripts/test_vlm_enricher.py`
- `test_enrich_chart_element`
- `test_enrich_table_element`
- `test_batch_enrichment`

### Verification

```bash
pytest tests/services/test_vlm.py tests/orchestrator/test_vlm_client.py tests/scripts/test_vlm_enricher.py -v

# Remote test
curl http://<vast-ai-ip>:8007/health
curl -X POST http://<vast-ai-ip>:8007/describe-chart \
  -H "Content-Type: application/json" \
  -d '{"image": "<base64_chart_image>", "prompt": "Phân tích biểu đồ kinh tế này."}'
```

---

## Summary

| Task | Description | Effort |
|------|-------------|--------|
| 1 | VLM Service Core | 4h |
| 2 | Dockerfile & Dependencies | 1h |
| 3 | VLM Client | 1.5h |
| 4 | VLM Enricher | 2h |
| 5 | Docker/Remote Config | 1h |
| 6 | Tests | 2h |
| **Total** | **6 tasks** | **10-12h** |
