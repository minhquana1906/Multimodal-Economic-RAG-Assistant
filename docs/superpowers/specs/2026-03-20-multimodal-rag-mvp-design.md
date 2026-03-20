# Multimodal Economic RAG Assistant — Phase 1 MVP Design Spec

## 1. Overview

A multimodal RAG system for Vietnamese economic and financial news, using the Hugging Face dataset `khoalnd/EconVNNews` (308K articles) as its knowledge base. Phase 1 delivers text-only RAG with hybrid retrieval, reranking, guardrails, and a public demo. The architecture is designed so Phases 2 (audio), 3 (image/chart), and 4 (PDF ingestion) plug in without restructuring.

**Language:** Vietnamese only (queries, prompts, answers, system messages).

**Key invariant:** Every modality (text, audio, image, PDF) reduces to a plain text query before entering the retrieval pipeline. Everything downstream of normalization is modality-agnostic.

## 2. System Architecture

### 2.1 High-Level Flow

```
User → Open WebUI → FastAPI Orchestrator → [Guard → Retriever → Reranker → LLM → Guard] → Answer + Citations
```

### 2.2 Service Topology (Phase 1)

| Service | Container | GPU? | Port (internal) | Framework |
|---------|-----------|------|-----------------|-----------|
| Open WebUI | `webui` | No | 8080 | Open WebUI |
| FastAPI Orchestrator | `orchestrator` | No | 8000 | FastAPI |
| Qdrant | `qdrant` | No | 6333/6334 | Qdrant |
| Embedding | `embedding` | Yes (local) | 8001 | sentence-transformers |
| Reranker | `reranker` | Yes (local) | 8002 | transformers |
| Guard | `guard` | Yes (local) | 8003 | transformers |
| LLM | N/A (remote) | Yes (remote) | 8004 | vLLM |
| Cloudflare Tunnel | `tunnel` | No | — | cloudflared |

Phase 2 adds: `asr` (port 8005, local GPU, `qwen-asr`).
Phase 3 adds: `vlm` (port 8006, remote GPU, `transformers`).

### 2.3 Model Placement

**Local RTX 3060 (12GB):**

| Model | VRAM (approx) | Library |
|-------|---------------|---------|
| Qwen3-Embedding-0.6B | ~1.2 GB | sentence-transformers |
| Qwen3-Reranker-0.6B | ~1.2 GB | transformers (AutoModelForCausalLM) |
| Qwen3Guard-Gen-0.6B | ~1.2 GB | transformers (AutoModelForCausalLM) |
| Qwen3-ASR-1.7B (Phase 2) | ~3.4 GB | qwen-asr |

**Remote GPU (vast.ai):**

| Model | VRAM (approx) | Library |
|-------|---------------|---------|
| Qwen3.5-4B | ~8 GB | vLLM |
| Qwen3-VL-4B-Instruct (Phase 3) | ~8 GB | transformers (Qwen3VLForConditionalGeneration) |

### 2.4 Networking

All services live on a private Docker bridge network (`rag-net`). Only the Cloudflare Tunnel container connects to the public internet, forwarding `quannguyen.works` → `webui:8080`. The orchestrator reaches remote vLLM instances via `LLM_URL` (SSH tunnel or Tailscale for security). No model service is directly exposed.

## 3. Data Ingestion & Indexing

### 3.1 Dataset

`khoalnd/EconVNNews` — 308,933 Vietnamese economic news articles.

| Column | Type | Description |
|--------|------|-------------|
| title | str | Article headline |
| url | str | Source URL |
| content | str | Full article body |
| published_date | str | e.g. "2020-09-01" |
| author | str | Author name |
| category | str | e.g. "Kinh te vi mo", "Kinh doanh" |
| source | str | e.g. "cafebiz" |

### 3.2 Chunking Strategy (Semantic)

Each article produces 1+ chunks:

| Chunk type | Content | Purpose |
|------------|---------|---------|
| `title_lead` | `title` + first paragraph of `content` | Broad topic matching |
| `body_paragraph` | Each subsequent paragraph (min 50 chars) | Granular fact retrieval |

All chunks from one article share a deterministic `article_id` (hash of `url`).

### 3.3 Qdrant Collection Schema

```
Collection: "econ_vn_news"

Vectors:
  - "dense": Qwen3-Embedding-0.6B output (1024 dimensions)
  - "sparse": BM25-based sparse vector

Payload:
  - article_id: str          # links chunks from same article
  - chunk_type: str           # "title_lead" | "body_paragraph"
  - chunk_index: int          # ordering within article
  - title: str                # original article title
  - url: str                  # source URL
  - published_date: str       # "2020-09-01"
  - author: str
  - category: str             # "Kinh te vi mo", etc.
  - source: str               # "cafebiz", etc.
  - text: str                 # actual chunk text
```

### 3.4 Hybrid Retrieval

1. Encode query → dense vector via Embedding service (with `prompt_name="query"`)
2. Encode query → sparse vector via BM25/sparse encoder
3. Qdrant `query` with `prefetch` for both dense and sparse, fused via Reciprocal Rank Fusion (RRF)
4. Return top-k (default k=20) candidates to reranker

### 3.5 Ingestion Script

`scripts/ingest.py` — one-off CLI script:
- Loads dataset via `datasets` library from Hugging Face
- Chunks articles per the semantic strategy
- Batches embeddings through the Embedding service (batch size ~256)
- Inserts into Qdrant with full payload
- Idempotent: checks if collection exists with expected point count before re-ingesting
- Estimated time: ~30–45 min for 308K articles on the 3060

## 4. FastAPI Orchestrator

### 4.1 Directory Structure

```
orchestrator/
├── main.py                  # FastAPI app, mounts routers
├── routers/
│   └── chat.py              # POST /v1/chat/completions (OpenAI-compatible)
├── services/
│   ├── guard.py             # Guard client
│   ├── embedder.py          # Embedding client
│   ├── retriever.py         # Qdrant hybrid search
│   ├── reranker.py          # Reranker client
│   └── llm.py               # LLM client (remote vLLM)
├── pipeline/
│   └── rag.py               # Full RAG pipeline assembly
├── models/
│   └── schemas.py           # Pydantic request/response models
└── config.py                # Env var configuration
```

### 4.2 RAG Pipeline

```
1. Extract user message text from OpenAI-format request
2. Guard check (input) → if unsafe: return apology immediately
3. Embed query → hybrid retrieve from Qdrant (top-20) → rerank (top-5)
4. Build prompt: system prompt + retrieved chunks with citations + user query
5. Generate answer via LLM (full response, not streaming)
6. Guard check (output):
   → safe: proceed
   → unsafe: retry once with safety feedback appended to prompt
   → still unsafe: return apology
7. Simulate streaming: chunk the final safe answer and yield SSE tokens
```

### 4.3 Open WebUI Integration

Open WebUI connects to FastAPI as an OpenAI-compatible backend via `POST /v1/chat/completions` with `stream: true`. Open WebUI is configured to point at `http://orchestrator:8000` as its only backend. It acts purely as a frontend shell — no plugins, no custom logic.

### 4.4 Citation Format

Appended to every answer:

```
---
Nguon tham khao:
[1] <title> — <source>, <published_date> — <url>
[2] ...
```

### 4.5 Configuration (env vars)

```
EMBEDDING_URL=http://embedding:8001
RERANKER_URL=http://reranker:8002
GUARD_URL=http://guard:8003
LLM_URL=http://<remote-ip>:8004
QDRANT_URL=http://qdrant:6333
QDRANT_COLLECTION=econ_vn_news
RETRIEVAL_TOP_K=20
RERANK_TOP_N=5
APOLOGY_MESSAGE="Xin loi, toi khong the tra loi cau hoi nay."
GUARD_TIMEOUT_S=10
LLM_TIMEOUT_S=60
EMBEDDING_TIMEOUT_S=15
RERANKER_TIMEOUT_S=15
```

## 5. Model Services — Implementation Details

### 5.1 Embedding Service

- **Model:** `Qwen/Qwen3-Embedding-0.6B`
- **Library:** `sentence-transformers`
- **API:** `POST /embed` → `{ "texts": [...], "is_query": bool }` → `{ "embeddings": [[...]] }`
- Query encoding uses `prompt_name="query"` (task instruction prepended automatically); document encoding uses no prompt.
- Output: 1024-dimensional dense vectors.

```python
from sentence_transformers import SentenceTransformer

model = SentenceTransformer("Qwen/Qwen3-Embedding-0.6B")

# Queries
query_embeddings = model.encode(queries, prompt_name="query")
# Documents
doc_embeddings = model.encode(documents)
```

### 5.2 Reranker Service

- **Model:** `Qwen/Qwen3-Reranker-0.6B`
- **Library:** `transformers` with `AutoModelForCausalLM`
- **API:** `POST /rerank` → `{ "query": "...", "passages": ["..."] }` → `{ "scores": [0.92, ...] }`
- **Not** a classical cross-encoder. Uses yes/no token logits to compute relevance:

```python
token_true_id = tokenizer.convert_tokens_to_ids("yes")
token_false_id = tokenizer.convert_tokens_to_ids("no")

# Format: <Instruct>: ...\n<Query>: ...\n<Document>: ...
logits = model(**inputs).logits[:, -1, :]
true_score = logits[:, token_true_id].exp().item()
false_score = logits[:, token_false_id].exp().item()
relevance_score = true_score / (true_score + false_score)
```

### 5.3 Guard Service

- **Model:** `Qwen/Qwen3Guard-Gen-0.6B`
- **Library:** `transformers` with `AutoModelForCausalLM`
- **API:** `POST /classify` → `{ "text": "...", "role": "input"|"output", "context": "..." }` → `{ "label": "safe"|"unsafe" }`

**Input moderation:** Format as `[{"role": "user", "content": text}]`, apply chat template, generate up to 128 tokens.

**Output moderation:** Format as `[{"role": "user", "content": prompt}, {"role": "assistant", "content": answer}]`.

Parse output: look for `Safety: Safe` / `Safety: Unsafe` / `Safety: Controversial`. Anything non-`Safe` is treated as unsafe.

### 5.4 LLM Service (Remote)

- **Model:** `Qwen/Qwen3.5-4B`
- **Serving:** `vllm serve Qwen/Qwen3.5-4B --port 8004 --reasoning-parser qwen3`
- Orchestrator calls vLLM's native `/v1/chat/completions` via OpenAI SDK.
- **Thinking mode disabled** for RAG answers: `extra_body={"chat_template_kwargs": {"enable_thinking": False}}`
- System prompt enforces Vietnamese-only answers with citation format.

### 5.5 ASR Service (Phase 2)

- **Model:** `Qwen/Qwen3-ASR-1.7B`
- **Library:** `qwen-asr` (`pip install qwen-asr`)
- **API:** `POST /transcribe` → multipart audio file → `{ "transcript": "...", "language": "vi" }`

```python
from qwen_asr import Qwen3ASRModel

model = Qwen3ASRModel.from_pretrained("Qwen/Qwen3-ASR-1.7B", dtype=torch.bfloat16, device_map="cuda:0")
results = model.transcribe(audio=audio_path, language=None)
```

Accepts: local file paths, URLs, base64 data, numpy arrays.

### 5.6 VLM Service (Phase 3, Remote)

- **Model:** `Qwen/Qwen3-VL-4B-Instruct`
- **Library:** `transformers` with `Qwen3VLForConditionalGeneration` + `AutoProcessor` (vLLM support not confirmed)
- **API:** `POST /analyze-image` → `{ "image_b64": "..." }` → `{ "summary": "..." }`
- Prompt instructs the model to extract structured facts from charts/images (not free-form VQA). The summary is used as a grounded text query for retrieval.

## 6. Docker Compose & Infrastructure

### 6.1 Local Docker Compose

All services on private bridge network `rag-net`:

```yaml
services:
  webui:         # Open WebUI, port 8080
  orchestrator:  # FastAPI, port 8000
  qdrant:        # Qdrant, ports 6333/6334, named volume
  embedding:     # GPU, port 8001
  reranker:      # GPU, port 8002
  guard:         # GPU, port 8003
  tunnel:        # Cloudflare Tunnel
  # Phase 2 (profile: "audio"):
  asr:           # GPU, port 8005
```

- GPU services get `deploy.resources.reservations.devices` with `driver: nvidia`
- Qdrant mounts `qdrant_storage:/qdrant/storage` for persistence
- `HUGGING_FACE_HUB_TOKEN` injected via `.env` for model downloads
- Phase 2/3 services use Docker Compose `profiles` (`audio`, `vision`) — opt-in via `docker compose --profile audio up`

### 6.2 Cloudflare Tunnel

```yaml
# tunnel/config.yml
ingress:
  - hostname: quannguyen.works
    service: http://webui:8080
  - service: http_status:404
```

Tunnel container uses `cloudflare/cloudflared:latest` with token via env var. Only `webui:8080` is reachable from the internet.

### 6.3 Remote GPU (vast.ai)

Not managed by Docker Compose. SSH into rented instance and run:

```bash
vllm serve Qwen/Qwen3.5-4B --port 8004 --reasoning-parser qwen3
# Phase 3:
vllm serve Qwen/Qwen3-VL-4B-Instruct --port 8006  # if vLLM supports it; else transformers wrapper
```

Orchestrator reaches them via `LLM_URL` / `VLM_URL` env vars.

### 6.4 Project Directory Layout

```
Multimodal-Economic-RAG-Assistant/
├── docker-compose.yml
├── .env.example
├── scripts/
│   └── ingest.py
├── orchestrator/
│   ├── Dockerfile
│   ├── main.py
│   ├── routers/
│   ├── services/
│   ├── pipeline/
│   ├── models/
│   └── config.py
├── services/
│   ├── embedding/
│   │   ├── Dockerfile
│   │   └── main.py
│   ├── reranker/
│   │   ├── Dockerfile
│   │   └── main.py
│   ├── guard/
│   │   ├── Dockerfile
│   │   └── main.py
│   └── asr/              # Phase 2
│       ├── Dockerfile
│       └── main.py
├── tunnel/
│   └── config.yml
└── docs/
    ├── diagrams/
    └── superpowers/specs/
```

## 7. Error Handling & Resilience

| Failure | Behavior |
|---------|----------|
| Guard service unreachable | Fail closed — treat as unsafe, return apology |
| Embedding service unreachable | Return HTTP 503 with user-facing error |
| Qdrant unreachable | Return HTTP 503 |
| Reranker unreachable | Fall back to raw retrieval order (skip reranking, log warning) |
| LLM service unreachable / timeout | Return HTTP 503 |
| LLM returns empty/malformed output | Return apology, log error |
| Guard output unparseable | Treat as unsafe, fail closed |

The apology message (`APOLOGY_MESSAGE`) is used for guardrail failures only. HTTP 503s are distinct so WebUI shows a "service unavailable" message. All timeouts configured via env vars.

## 8. Extensibility

### Phase 2 — Audio

Add `POST /v1/audio/transcriptions` to the orchestrator. It calls the ASR service, gets a transcript, and re-routes it through the existing RAG pipeline. The `/v1/chat/completions` endpoint stays untouched.

### Phase 3 — Image/Chart

The orchestrator detects image attachments in the OpenAI-format message (base64 in `content` array), calls the VLM service for structured text extraction, then runs the standard RAG pipeline on the resulting text query. No changes to retrieval, reranking, or LLM layers.

### Phase 4 — PDF/Docling

Add `scripts/ingest_pdf.py` using Docling to extract text. Chunk using the same semantic strategy. Upsert into the same Qdrant collection with `source_type: "pdf"` in the payload. The retrieval and generation pipeline requires zero changes.

### Design Invariant

Everything downstream of `normalize_query(text) → str` is modality-agnostic. ASR, VLM, and future Docling extraction all reduce to text before entering retrieval.
