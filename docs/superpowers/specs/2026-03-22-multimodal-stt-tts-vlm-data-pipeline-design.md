# Multimodal RAG Extension — STT, TTS, VLM & Economic Data Pipeline Design Spec

**Date:** 2026-03-22
**Status:** Reviewed (1 pass, all issues addressed in §13)
**Depends on:** Phase 1 MVP (complete), Central Config + Loguru (complete)
**Scope:** STT (ASR), TTS, VLM services + economic data crawl/ingest pipeline

---

## 1. Overview

Extend the existing text-only RAG system into a full multimodal assistant by adding three new model services (ASR, TTS, VLM) and building a comprehensive economic data pipeline that crawls, parses, and indexes content from diverse Vietnamese sources — including charts, tables, and figures found in PDFs and web pages.

**Design invariant (preserved from Phase 1):** Every modality (text, audio, image) normalizes to a plain text query before entering the retrieval pipeline. Everything downstream of normalization is modality-agnostic.

**Key additions:**
- **STT:** Qwen3-ASR-1.7B (local GPU, on-demand loading)
- **TTS:** VieNeu-1000h (local GPU, on-demand loading)
- **VLM:** Qwen3-VL-4B-Instruct (remote vast.ai)
- **Data Pipeline:** Multi-source crawlers + PDF parser + VLM enrichment + multi-collection Qdrant

---

## 2. System Architecture

### 2.1 High-Level Flow

```
INPUT NORMALIZATION:
  Audio ──► ASR Service ──► Text
  Text  ──────────────────► Text  ──► Normalized Query
  Image ──► VLM Service ──► Text

EXISTING RAG PIPELINE (untouched):
  Guard → Embed → Retrieve(multi-col) → Rerank → Web Fallback → LLM → Guard

OUTPUT FORMATTING:
  Text Response ──────────────────► Text
  Text Response ──► TTS Service ──► Audio
```

### 2.2 Service Topology (Extended)

| Service | Port | GPU | Location | Model | VRAM | New? |
|---------|------|-----|----------|-------|------|------|
| Embedding | 8001 | Yes | Local | Qwen3-Embedding-0.6B | ~1.2 GB | No |
| Reranker | 8002 | Yes | Local | Qwen3-Reranker-0.6B | ~1.2 GB | No |
| Guard | 8003 | Yes | Local | Qwen3Guard-Gen-0.6B | ~1.2 GB | No |
| LLM | 8004 | Yes | Remote | Qwen3.5-4B | ~8 GB | No |
| **ASR** | **8005** | **Yes** | **Local** | **Qwen3-ASR-1.7B** | **~3.4 GB** | **Yes** |
| **TTS** | **8006** | **Yes** | **Local** | **VieNeu-1000h** | **~2-3 GB** | **Yes** |
| **VLM** | **8007** | **Yes** | **Remote** | **Qwen3-VL-4B-Instruct** | **~8 GB** | **Yes** |

### 2.3 VRAM Budget (Local RTX 3060, 12 GB)

Always loaded (3 services): 3.6 GB
On-demand ASR: +3.4 GB → peak 7.0 GB
On-demand TTS: +2-3 GB → peak 5.6-6.6 GB
On-demand ASR+TTS: +5.4-6.4 GB → peak 9.0-10.0 GB
CUDA overhead: ~1-2 GB
**Total peak: ~10-12 GB** — fits RTX 3060

### 2.4 On-Demand GPU Management Pattern

ASR and TTS services use a shared pattern for VRAM conservation:

1. Model is **not loaded** at container startup
2. First request triggers model loading (~5-8s cold start)
3. Subsequent requests use the loaded model instantly
4. After `MODEL_IDLE_TIMEOUT` (default 300s) with no requests, model is unloaded
5. `torch.cuda.empty_cache()` is called after unload to free VRAM

This ensures embedding/reranker/guard always have VRAM, and ASR/TTS only consume VRAM when actively needed.

---

## 3. ASR Service (Speech-to-Text)

### 3.1 Service Design

- **Model:** `Qwen/Qwen3-ASR-1.7B` via `qwen-asr` package
- **Location:** `services/asr/asr_app.py`
- **Port:** 8005
- **Docker profile:** `audio`

### 3.2 API

```
POST /transcribe
  Request: multipart/form-data
    - file: audio (WAV, MP3, WebM, OGG, FLAC)
    - language: str = "vi"
  Response: {
    "text": "transcribed Vietnamese text",
    "language": "vi",
    "duration_seconds": 3.2
  }

GET /health
  Response: {"status": "ok"|"loading"|"idle", "model_loaded": bool}
```

### 3.3 Audio Format Handling

Accept any common format. Use `torchaudio` or `ffmpeg` subprocess to convert to 16kHz mono WAV before passing to the model. Max audio duration: 60 seconds (configurable via `ASR_MAX_DURATION_S`).

### 3.4 Orchestrator Integration

New `ASRClient` in `api/orchestrator/services/asr.py`:
- `async def transcribe(audio_bytes: bytes, language: str = "vi") -> str`
- Sends multipart POST to ASR service
- Returns transcribed text
- On error: raises `ASRError` (orchestrator returns 503)

---

## 4. TTS Service (Text-to-Speech)

### 4.1 Service Design

- **Model:** VieNeu-1000h (Vietnamese neural TTS)
- **Location:** `services/tts/tts_app.py`
- **Port:** 8006
- **Docker profile:** `audio`

### 4.2 API

```
POST /synthesize
  Request: {
    "text": "Vietnamese text to speak",
    "speed": 1.0,
    "sample_rate": 22050
  }
  Response: audio/wav binary
  Headers: Content-Type: audio/wav, X-Duration-Seconds: 2.8

POST /stream
  Same request body
  Response: chunked audio/wav (Server-Sent Events with base64 audio segments)

GET /health
  Response: {"status": "ok"|"loading"|"idle", "model_loaded": bool}
```

### 4.3 Vietnamese Text Preprocessing

Before synthesis, text must be preprocessed:

1. **Number normalization:** "6.5%" → "sáu phẩy năm phần trăm"
2. **Abbreviation expansion:** "GDP" → "Tổng sản phẩm quốc nội", "NHNN" → "Ngân hàng Nhà nước"
3. **Sentence splitting:** Long text split at sentence boundaries for chunked synthesis
4. **Special character handling:** Remove markdown, citations like [1], URLs

An abbreviation dictionary for Vietnamese economic terms is maintained in `services/tts/abbreviations.py`.

### 4.4 Orchestrator Integration

New `TTSClient` in `api/orchestrator/services/tts.py`:
- `async def synthesize(text: str, speed: float = 1.0) -> bytes`
- Returns WAV audio bytes
- On error: returns None (text-only fallback, no audio in response)

---

## 5. VLM Service (Vision-Language Model)

### 5.1 Service Design

- **Model:** `Qwen/Qwen3-VL-4B-Instruct` via `transformers`
- **Location:** `services/vlm/vlm_app.py`
- **Port:** 8007
- **Deployment:** Remote vast.ai (alongside vLLM server)

### 5.2 API

```
POST /analyze
  Request: {"image": "<base64>", "prompt": "...", "max_tokens": 1024}
  Response: {
    "description": "detailed text description",
    "structured_data": { ... }  // optional
  }

POST /extract-table
  Request: {"image": "<base64>", "prompt": "Extract this table as markdown."}
  Response: {
    "markdown_table": "| Col1 | Col2 |\n|------|------|\n| ... |",
    "raw_text": "plain text representation"
  }

POST /describe-chart
  Request: {"image": "<base64>", "prompt": "Analyze this economic chart."}
  Response: {
    "description": "Chart shows GDP growth from 2019-2023...",
    "trend": "increasing|decreasing|stable|mixed",
    "key_insights": ["insight1", "insight2"]
  }

GET /health
```

### 5.3 Dual-Purpose Usage

The VLM serves two roles:

1. **Runtime (query-time):** User uploads a chart image → VLM describes it → description becomes the RAG query
2. **Ingestion (offline):** PDF/web charts and tables → VLM generates text descriptions → stored in Qdrant as searchable chunks

### 5.4 Dual Storage Strategy for Visual Elements

When processing charts/tables from documents:

1. **Image extracted** from PDF/web page → saved to `data/images/{doc_hash}_{page}_{idx}.png`
2. **VLM generates text description** → becomes the chunk text (searchable via embedding)
3. **Both are stored in Qdrant:**
   - `text` field: VLM-generated description (embedded, searchable)
   - `image_path` field in payload: path to original image
   - `structured_data` field: extracted data points (for tables: markdown; for charts: trend + insights)
   - `chunk_type`: `"table"` or `"chart"`

4. **At retrieval time:** When a chunk with `image_path` is retrieved, the response includes both the text description and a reference to the original image, allowing the frontend to display the visual alongside the answer.

### 5.5 Orchestrator Integration

New `VLMClient` in `api/orchestrator/services/vlm.py`:
- `async def analyze(image_b64: str, prompt: str) -> dict`
- `async def extract_table(image_b64: str) -> dict`
- `async def describe_chart(image_b64: str) -> dict`
- On error: returns empty dict (text-only fallback)

---

## 6. Data Pipeline

### 6.1 Architecture Overview

Four-stage offline batch pipeline:

```
STAGE 1: CRAWL & COLLECT
  Web crawlers (httpx+BS4) → HTML articles
  PDF downloaders → PDF files
  Academic scrapers → PDF papers
  → Raw data store: data/raw/{source}/{date}/{hash}.{html|pdf}

STAGE 2: PARSE & EXTRACT
  HTML → trafilatura + BS4 → structured text
  PDF → pdfplumber + PyMuPDF → text blocks + image regions
  Images → visual_extractor → chart/table PNGs
  → Parsed data store: data/parsed/{source}/{hash}.json

STAGE 3: CHUNK & ENRICH
  Text → extended chunker (title_lead, body_paragraph, section_content)
  Charts → VLM /describe-chart → chart_description chunks
  Tables → VLM /extract-table → table_content chunks
  → Unified chunks with consistent schema

STAGE 4: EMBED & INDEX
  Chunks → Embedding service (dense 1024d)
  Chunks → BM25 tokenizer (sparse, underthesea)
  → Qdrant collection: "econ_knowledge"
```

### 6.2 Data Sources

#### Tier 1: Vietnamese Economic News (Web scraping)

| Source | Method | Quality |
|--------|--------|---------|
| VnExpress/Kinh doanh | httpx + BS4 | Medium |
| CafeF | httpx + BS4 | High |
| CafeBiz | httpx + BS4 | Medium |
| VietnamBiz | httpx + BS4 | High |
| VnEconomy | httpx + BS4 | Medium |
| Bao Dau Tu | httpx + BS4 | Medium |
| TheSaigonTimes | httpx + BS4 | High |

Rate limiting: 1-2 req/sec per domain. Robots.txt respected. Dedup by URL hash.

#### Tier 2: Corporate Reports (PDF download)

| Source | Method | Quality |
|--------|--------|---------|
| CafeF financial reports | Direct PDF download | High |
| VietStock reports | Direct PDF download | High |
| SSC regulatory filings | Direct PDF download | High |
| Company annual reports | Selenium + download | High |

Heavy tables, financial statements, charts. VLM processing required.

#### Tier 3: Academic & Educational (Mixed)

| Source | Method | Quality |
|--------|--------|---------|
| Google Scholar (Vietnam economics) | Selenium | High |
| UEH/NEU/FTU repositories | httpx | High |
| SSRN (Vietnam tag) | httpx | High |

#### Tier 4: Knowledge Blogs & Educational Sites (Web scraping)

| Source | Method | Quality |
|--------|--------|---------|
| VietnamFinance | httpx + BS4 | Medium |
| StockBiz | httpx + BS4 | Medium |
| TapchiTaiChinh | httpx + BS4 | Medium |
| Kinh tế Sài Gòn | httpx + BS4 | Medium |
| Economics blogs/explainers | httpx + BS4 | Basic |
| University econ course materials | httpx | Basic-Medium |

### 6.3 PDF Processing Pipeline

```
PDF File
  │
  ▼
Step 1: Layout Analysis (pdfplumber + PyMuPDF)
  Per page detect: text blocks, image regions, table regions, headers/footers
  │
  ├── Text Blocks ──► Step 2a: Text Extraction
  │                    • Section detection (heading fonts)
  │                    • Paragraph merging
  │                    • Citation/reference stripping
  │
  └── Visual Regions ──► Step 2b: Visual Extraction
                          • Extract image from PDF (PyMuPDF fitz)
                          • Save to data/images/
                          │
                          ├── Has grid lines? → Table → VLM /extract-table
                          └── No grid lines? → Chart → VLM /describe-chart
  │
  ▼
Step 3: Unified Document Object
  { text_sections, tables (markdown + raw), charts (description + insights), metadata }
```

### 6.4 Extended Chunking Strategy

| Document Type | Chunk Types Produced |
|---------------|---------------------|
| Web Article | `title_lead`, `body_paragraph` (existing) |
| PDF Report | `title_lead`, `section_content`, `table_content`, `chart_description` |
| Academic Paper | `title_lead` (title+abstract), `section_content`, `table_content`, `chart_description` |
| Knowledge Blog | `title_lead`, `body_paragraph`, `definition` (Q&A patterns, glossary items) |

Chunk size limits: min 50 chars (merge if shorter), max 1500 chars (split at sentence boundary). Table chunks have no max limit. Chart description chunks are typically 200-500 chars from VLM output.

### 6.5 Multi-Collection Qdrant Schema

**Existing collection (unchanged):**
```
Collection: "econ_vn_news"
  Vectors: dense(1024, COSINE) + sparse(BM25)
  Payload: article_id, chunk_type, chunk_index, title, url, published_date, category, source, text
```

**New collection:**
```
Collection: "econ_knowledge"
  Vectors: dense(1024, COSINE) + sparse(BM25)
  Payload:
    doc_id: str              # deterministic hash of source URL/path
    chunk_type: str          # title_lead | body_paragraph | section_content | table_content | chart_description | definition
    chunk_index: int
    title: str
    url: str
    source: str              # "cafef", "vnexpress", "ueh_repository", etc.
    published_date: str
    category: str
    doc_type: str            # "article" | "report" | "paper" | "textbook" | "blog"
    text: str                # searchable text content
    image_path: str | null   # path to chart/table image (if visual element)
    structured_data: dict | null  # markdown table or chart insights
    source_quality: str      # "high" | "medium" | "basic"
```

### 6.6 Multi-Collection Retrieval

At query time, the orchestrator searches both collections in parallel, merges results, then reranks:

```
Query → Embed (1024d)
  │
  ├── econ_vn_news: prefetch dense top-20 + sparse top-20, RRF → top-10
  ├── econ_knowledge: prefetch dense top-20 + sparse top-20, RRF → top-10
  │
  ▼
Merge (20 candidates) → Reranker → top-5
  │
  ▼
Continue existing pipeline (web fallback → generate → ...)
```

The retriever client is updated to accept a `collections` parameter (list of collection names). Reranker operates on text only and is collection-agnostic.

---

## 7. Extended LangGraph Pipeline

### 7.1 Extended RAGState

```python
class RAGState(TypedDict):
    # Input normalization
    input_modality: str             # "text" | "audio" | "image"
    audio_file: bytes | None        # raw audio (if audio input)
    image_data: str | None          # base64 image (if image input)
    query: str                      # normalized text query

    # Existing pipeline state (untouched)
    input_safe: bool
    embeddings: list[float]
    retrieved_docs: list[dict]
    reranked_docs: list[dict]
    web_results: list[dict]
    final_context: list[dict]
    answer: str
    output_safe: bool
    citations: list[dict]
    error: str | None

    # Output formatting
    response_format: str            # "text" | "audio" | "text+audio"
    audio_response: bytes | None    # TTS output WAV bytes
```

### 7.2 New Nodes

**normalize_input:** Detects input modality from request. Sets `input_modality`.

**asr_node:** Calls ASR service with `audio_file`. Sets `query` to transcribed text. Skipped if `input_modality != "audio"`.

**vlm_query_node:** Calls VLM `/analyze` with `image_data`. Sets `query` to VLM description. Skipped if `input_modality != "image"`.

**format_output:** After citations node. If `response_format` includes audio, calls TTS service. Sets `audio_response`.

### 7.3 Updated Graph Edges

```
START → normalize_input
  → [route_by_modality]
    → "audio": asr_node → input_guard
    → "image": vlm_query_node → input_guard
    → "text": input_guard (direct)
  → ... existing pipeline unchanged ...
  → citations → format_output → END
```

### 7.4 Chat Endpoint Updates

`POST /v1/chat/completions` extended to accept:

- **Audio input:** Multipart form with audio file + messages JSON
- **Image input:** Base64 image in OpenAI-format content array `[{"type": "image_url", "image_url": {"url": "data:image/png;base64,..."}}]`
- **Audio output request:** `response_format: "audio"` or `"text+audio"` in request body

Response extended with optional `audio` field containing base64-encoded WAV.

---

## 8. Directory Structure (New Files)

```
services/
├── asr/                          # NEW
│   ├── asr_app.py               # FastAPI + Qwen3-ASR-1.7B
│   ├── Dockerfile
│   └── requirements.txt
├── tts/                          # NEW
│   ├── tts_app.py               # FastAPI + VieNeu-1000h
│   ├── abbreviations.py         # Vietnamese economic abbreviation dict
│   ├── text_preprocessor.py     # Number/abbreviation normalization
│   ├── Dockerfile
│   └── requirements.txt
├── vlm/                          # NEW
│   ├── vlm_app.py               # FastAPI + Qwen3-VL-4B-Instruct
│   ├── Dockerfile
│   └── requirements.txt

api/orchestrator/services/
├── asr.py                        # NEW: ASRClient
├── tts.py                        # NEW: TTSClient
├── vlm.py                        # NEW: VLMClient

scripts/
├── crawlers/                     # NEW
│   ├── __init__.py
│   ├── base.py                  # BaseCrawler ABC
│   ├── config.py                # Source registry
│   ├── web_crawler.py           # HTML article crawler
│   ├── pdf_crawler.py           # PDF downloader
│   ├── academic_crawler.py      # Scholar/repository crawler
│   └── sources/                 # Per-source CSS selectors & configs
│       ├── vnexpress.py
│       ├── cafef.py
│       ├── vietnambiz.py
│       └── ...
├── parsers/                      # NEW
│   ├── __init__.py
│   ├── html_parser.py           # trafilatura + BS4
│   ├── pdf_parser.py            # pdfplumber + PyMuPDF
│   └── visual_extractor.py      # chart/table image extraction
├── pipeline/                     # NEW
│   ├── __init__.py
│   ├── orchestrator.py          # Main: crawl → parse → chunk → embed → index
│   ├── vlm_enricher.py          # Batch VLM processing for charts/tables
│   └── multi_collection.py      # Qdrant multi-collection management
└── ingest_crawled.py             # NEW: CLI for crawled data ingestion

data/                             # NEW (gitignored)
├── raw/{source}/                # Raw crawled HTML/PDF
├── parsed/{source}/             # Parsed JSON documents
└── images/                      # Extracted chart/table PNGs
```

---

## 9. Docker Compose Profiles

```yaml
# Profile: default (no flag) — existing text-only RAG
# qdrant + embedding + reranker + guard + orchestrator + webui + tunnel

# Profile: --profile audio — adds voice I/O
# + asr (port 8005, local GPU) + tts (port 8006, local GPU)

# Profile: --profile vision — adds image understanding
# + vlm config (remote, env vars only)

# Profile: --profile full — everything
# default + audio + vision

# Profile: --profile ingest — existing HF dataset ingest
# qdrant + embedding + ingest script

# Profile: --profile crawl — crawled data ingest
# qdrant + embedding + vlm + crawl-ingest script
```

---

## 10. Configuration Extensions

New environment variables (following existing `env_nested_delimiter="__"` pattern):

```env
# ASR Service
SERVICES__ASR_URL=http://asr:8005
SERVICES__ASR_MODEL=Qwen/Qwen3-ASR-1.7B
SERVICES__ASR_TIMEOUT=30.0
SERVICES__ASR_MAX_DURATION_S=60
SERVICES__ASR_IDLE_TIMEOUT=300

# TTS Service
SERVICES__TTS_URL=http://tts:8006
SERVICES__TTS_MODEL=VieNeu-1000h
SERVICES__TTS_TIMEOUT=30.0
SERVICES__TTS_SPEED=1.0
SERVICES__TTS_SAMPLE_RATE=22050
SERVICES__TTS_IDLE_TIMEOUT=300

# VLM Service
SERVICES__VLM_URL=http://<remote-ip>:8007
SERVICES__VLM_MODEL=Qwen/Qwen3-VL-4B-Instruct
SERVICES__VLM_TIMEOUT=60.0
SERVICES__VLM_MAX_TOKENS=1024

# Multi-collection retrieval
RAG__COLLECTIONS=["econ_vn_news","econ_knowledge"]
RAG__PER_COLLECTION_TOP_K=10

# Data pipeline
CRAWL__DATA_DIR=./data
CRAWL__RATE_LIMIT_PER_SEC=1.5
CRAWL__RESPECT_ROBOTS_TXT=true
CRAWL__MAX_PAGES_PER_SOURCE=1000
```

---

## 11. Error Handling

| Failure | Behavior |
|---------|----------|
| ASR service unreachable | Return 503 "Voice input temporarily unavailable" |
| ASR returns empty transcription | Return 400 "Could not transcribe audio" |
| TTS service unreachable | Fall back to text-only response (no audio) |
| VLM service unreachable (runtime) | Return 503 "Image analysis temporarily unavailable" |
| VLM service unreachable (ingestion) | Skip visual element, log warning, continue with text-only chunks |
| Audio file too large / too long | Return 413 with max size/duration info |
| Unsupported audio format | Return 415 with list of supported formats |
| Image too large | Resize before sending to VLM (max 1024x1024) |
| Crawler rate limited / blocked | Back off exponentially, skip source after 3 failures |
| PDF parsing failure | Log error, skip document, continue pipeline |
| econ_knowledge collection empty | Retrieval falls back to econ_vn_news only |

---

## 12. Testing Strategy

Each new component gets unit tests with mocked dependencies:

| Component | Test File | Key Tests |
|-----------|-----------|-----------|
| ASR service | `tests/services/test_asr.py` | health, transcribe, format handling, idle unload |
| TTS service | `tests/services/test_tts.py` | health, synthesize, Vietnamese preprocessing |
| VLM service | `tests/services/test_vlm.py` | health, analyze, extract-table, describe-chart |
| ASR client | `tests/orchestrator/test_asr_client.py` | transcribe, error handling |
| TTS client | `tests/orchestrator/test_tts_client.py` | synthesize, fallback |
| VLM client | `tests/orchestrator/test_vlm_client.py` | analyze, error handling |
| Multimodal pipeline | `tests/orchestrator/test_multimodal_pipeline.py` | audio→text, image→text, format_output |
| Web crawler | `tests/scripts/test_crawlers.py` | crawl, dedup, rate limit |
| PDF parser | `tests/scripts/test_parsers.py` | text extract, table detect, image extract |
| VLM enricher | `tests/scripts/test_vlm_enricher.py` | chart describe, table extract |
| Extended chunker | `tests/scripts/test_chunker_extended.py` | section_content, table_content, chart_description |
| Data pipeline | `tests/scripts/test_data_pipeline.py` | end-to-end crawl→ingest flow |

**Estimated: ~12 new test files, ~40-50 new test functions.**

---

## 13. Spec Review Fixes (from reviewer pass 1)

### Fix 1: Port Numbering — Phase 1 Deviation Acknowledged

Phase 1 spec (§5.6) reserved port 8006 for VLM. This spec reassigns:
- TTS → port 8006 (new)
- VLM → port 8007 (moved)

**Rationale:** TTS is local (needs Docker port mapping), VLM is remote (only needs `VLM_URL` env var, no Docker port). Giving TTS the lower port makes the local service range contiguous (8001-8006). The commented-out ASR block in `docker-compose.yml` should be updated to reflect the final port plan.

### Fix 2: Multi-Collection Retrieval — Full Scope Clarified

The multi-collection change (Phase 3A Task 6) is larger than initially scoped. Full impact:

1. **`RetrieverClient`** — Accept `collections: list[str]` param, search each in parallel via `asyncio.gather`
2. **`config.py`** — Add `RAG__COLLECTIONS: list[str]` and `RAG__PER_COLLECTION_TOP_K: int` to `RAGConfig`
3. **`main.py`** — Construct `RetrieverClient` with collection list from config
4. **`rag.py` `retrieve_node`** — Pass collection list to retriever
5. **Payload extraction** — Include `image_path`, `structured_data`, `chunk_type`, `doc_type` from Qdrant payload (needed for visual chunk rendering)

Phase 3A Task 6 effort estimate increased from 1.5h to 3h.

### Fix 3: Sparse Vector Query-Time Gap — Pre-Requisite Task Added

The current Phase 1 pipeline only passes dense vectors at query time. Sparse (BM25) vectors are generated at ingestion but **never generated for queries**. This means retrieval is currently dense-only, not hybrid.

**Required fix (Phase 1.5 or early Phase 3A):**
1. Add `underthesea` + `fastembed` to orchestrator requirements
2. Add BM25 query encoding in `embed_node` (or a new `sparse_embed_node`)
3. Pass both dense + sparse vectors to `RetrieverClient.hybrid_search()`
4. This must be done before multi-collection retrieval work begins

Added as Phase 3A Task 0 (pre-requisite): "Wire sparse BM25 vectors into query-time pipeline" (2h).

### Fix 4: Message.content Type — Union Type for Multimodal

The existing `Message.content: str` must change to `str | list[dict]` to support OpenAI-format image content arrays:

```python
class ContentPart(BaseModel):
    type: str  # "text" | "image_url"
    text: str | None = None
    image_url: dict | None = None  # {"url": "data:image/png;base64,..."}

class Message(BaseModel):
    role: Literal["system", "user", "assistant"]
    content: str | list[ContentPart]
```

This is a breaking schema change. Added explicitly to Phase 3A Task 7 scope. The `chat.py` router must extract text content from both string and array formats.

### Fix 5: Audio Input — Dual Endpoint Approach

The audio upload mechanism is resolved as follows:

- **`POST /v1/chat/completions`** — JSON only. Accepts text and image (base64 in content array). This is the existing OpenAI-compatible endpoint.
- **`POST /v1/audio/transcriptions`** — Multipart form. Accepts audio file upload. Returns transcribed text. Client then sends text via the standard chat endpoint.

This follows the OpenAI API pattern (separate `/audio/transcriptions` endpoint) and avoids mixing multipart and JSON in a single endpoint. The orchestrator internally routes audio through ASR → text → existing pipeline.

Alternatively, for a seamless single-request voice flow, clients can use a new convenience endpoint:
- **`POST /v1/chat/audio`** — Multipart form with `audio` file + `messages` JSON string field + `response_format`. Internally calls ASR, then the standard pipeline, then TTS if requested.

### Fix 6: VRAM — Sequential ASR→TTS in Voice Flow

To avoid razor-thin VRAM (11.5/12 GB) during voice flows:

In `format_output` node, before calling TTS, explicitly request ASR model unload:
```
format_output:
  1. POST asr:8005/unload (if ASR was used in this request)
  2. Wait for VRAM release
  3. POST tts:8006/synthesize
```

Both ASR and TTS services expose `POST /unload` endpoint for explicit model release. This ensures only one on-demand model is loaded at a time during the voice flow, keeping peak VRAM at ~7 GB (3.6 always-on + 3.4 ASR or 3.0 TTS).

### Fix 7: On-Demand Loading — Thread-Safe Lock

The OnDemandModel class must include a loading lock to prevent concurrent load attempts:

```python
class OnDemandModel:
    def __init__(self):
        self._model = None
        self._load_lock = asyncio.Lock()
        self._idle_timer = None

    async def get_model(self):
        async with self._load_lock:
            if self._model is None:
                self._model = await self._load()
            self._reset_idle_timer()
            return self._model
```

This is added to both Phase 2A Task 1 (ASR) and Phase 2B Task 2 (TTS).

### Fix 8: Conditional Service Population

When running without `--profile audio`, ASR/TTS services are not available. The orchestrator must handle this:

```python
# In main.py lifespan:
services.asr = ASRClient(config.services.asr_url, ...) if config.services.asr_url else None
services.tts = TTSClient(config.services.tts_url, ...) if config.services.tts_url else None
services.vlm = VLMClient(config.services.vlm_url, ...) if config.services.vlm_url else None
```

Graph nodes check for None before calling:
- `asr_node`: if `services.asr is None`, return error "ASR service not configured"
- `vlm_query_node`: if `services.vlm is None`, return error "VLM service not configured"
- `format_output`: if `services.tts is None`, skip audio output silently

The `normalize_input` node should also validate: if modality is "audio" but ASR is not configured, return 501.

### Fix 9: Retriever Payload Extraction — Extended Fields

The `RetrieverClient.hybrid_search()` return format must include new fields:

```python
return [
    {
        "id": str(r.id),
        "text": r.payload.get("text", ""),
        "source": r.payload.get("source", ""),
        "title": r.payload.get("title", ""),
        "score": r.score,
        # NEW fields for visual chunks:
        "image_path": r.payload.get("image_path"),
        "structured_data": r.payload.get("structured_data"),
        "chunk_type": r.payload.get("chunk_type", "body_paragraph"),
        "doc_type": r.payload.get("doc_type", "article"),
        "collection": collection_name,  # which collection this came from
    }
    for r in results
]
```

The `citations_node` should render visual chunks differently (include image reference).

---

## 14. Open Questions / Future Decisions

1. **Open WebUI audio playback:** Open WebUI doesn't natively support audio in responses. Options: custom plugin, separate audio endpoint, or embed audio as download link in markdown.
2. **Selenium vs Playwright:** For academic crawlers, playwright (async-native) may be better than Selenium. Decide during Phase 2C Task 4 research.
3. **VieNeu-1000h validation:** Model loading API and actual VRAM must be confirmed in Phase 2B Task 1 research spike. Fallback: `vinai/PhoTTS` or `espnet` Vietnamese model.
4. **Docker profile granularity:** Current `audio` profile bundles ASR+TTS. Consider separate `asr` and `tts` profiles for more flexibility.
