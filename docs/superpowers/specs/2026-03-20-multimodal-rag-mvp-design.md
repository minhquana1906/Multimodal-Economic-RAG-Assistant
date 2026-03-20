# Multimodal Economic RAG Assistant — Phase 1 MVP Design Spec

## 1. Overview

A multimodal RAG system for Vietnamese economic and financial news, using the Hugging Face dataset `khoalnd/EconVNNews` (308K articles) as its knowledge base. Phase 1 delivers text-only RAG with hybrid retrieval, reranking, guardrails, and a public demo. The architecture is designed so Phases 2 (audio), 3 (image/chart), and 4 (PDF ingestion) plug in without restructuring.

**Language:** Vietnamese only (queries, prompts, answers, system messages).

**Key invariant:** Every modality (text, audio, image, PDF) reduces to a plain text query before entering the retrieval pipeline. Everything downstream of normalization is modality-agnostic.

## 2. System Architecture

### 2.1 High-Level Flow

```
User → Open WebUI → FastAPI Orchestrator → [Guard → Retriever → Reranker → (Web Fallback?) → LLM → Guard] → Answer + Citations
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
| Tavily Web Search | N/A (external API) | No | — | tavily-python |

Phase 2 adds: `asr` (port 8005, local GPU, `qwen-asr`).
Phase 3 adds: `vlm` (port 8006, remote GPU, `transformers`).

### 2.3 Model Placement

> **VRAM note:** Estimates below reflect weight size only. Allow ~1.5–2 GB additional overhead for CUDA contexts and inference buffers (each container creates its own CUDA context at ~300–500 MB). Phase 1 realistic total: ~5–6 GB on 12 GB. Phase 1+2 combined: ~9–10 GB — feasible but tight. Consider `CUDA_VISIBLE_DEVICES` pinning or sequential loading if OOM occurs.

**Local RTX 3060 (12GB):**

| Model | VRAM (weights) | Library |
|-------|----------------|---------|
| Qwen3-Embedding-0.6B | ~1.2 GB | sentence-transformers |
| Qwen3-Reranker-0.6B | ~1.2 GB | transformers (AutoModelForCausalLM) |
| Qwen3Guard-Gen-0.6B | ~1.2 GB | transformers (AutoModelForCausalLM) |
| Qwen3-ASR-1.7B (Phase 2) | ~3.4 GB | qwen-asr |

**Remote GPU (vast.ai):**

| Model | VRAM (weights) | Library |
|-------|----------------|---------|
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

**Short paragraph handling:** Paragraphs under 50 characters are merged with the preceding paragraph. If the first body paragraph is under 50 chars, it is merged into the `title_lead` chunk.

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
2. Encode query → sparse vector via Qdrant's built-in BM25 (see §3.4.1)
3. Qdrant `query` with `prefetch` for both dense and sparse, fused via Reciprocal Rank Fusion (RRF) using Qdrant's default `k=60`
4. Prefetch counts: top-40 dense + top-40 sparse → RRF fusion → return top-20 candidates to reranker

#### 3.4.1 Sparse Retrieval

**Method:** Qdrant's built-in BM25 sparse vector index via `qdrant_client`. When creating the collection, configure a `sparse_vectors` named `"sparse"` with `models.SparseVectorParams(modifier=models.Modifier.IDF)`.

**Vietnamese tokenization:** Use `underthesea` library (`word_tokenize`) to segment Vietnamese text before BM25 tokenization. Vietnamese is not whitespace-delimited at the word level, so raw splitting produces incorrect terms. The tokenizer runs both at ingestion time (on each chunk's text) and at query time (on the user query).

**Ingestion flow:**
1. For each chunk: segment text with `underthesea.word_tokenize(text)`
2. Build sparse vector using `fastembed` with the `Qdrant/bm25` model (which accepts pre-tokenized text)
3. Upload both dense and sparse vectors together in a single upsert batch

**Query flow:**
1. Segment query with `underthesea.word_tokenize(query)`
2. Encode via the same `Qdrant/bm25` sparse model
3. Pass sparse vector to Qdrant prefetch alongside the dense vector

### 3.5 Ingestion Script

`scripts/ingest.py` — one-off CLI script:
- Loads dataset via `datasets` library from Hugging Face
- Chunks articles per the semantic strategy
- Batches embeddings through the Embedding service (batch size ~256)
- Generates sparse vectors via `fastembed` with `Qdrant/bm25` model (Vietnamese text pre-tokenized with `underthesea`)
- Inserts both dense + sparse vectors into Qdrant with full payload
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

**Conversation history:** Only the last `user` message is used as the retrieval query. Previous conversation turns are discarded (stateless RAG for Phase 1 MVP).

**Pipeline steps:**

```
1. Extract the last user message from the OpenAI-format messages array
2. Guard check (input) → if unsafe: return apology immediately
3. Embed query → hybrid retrieve from Qdrant (top-20) → rerank (top-5)
4. Evaluate reranked results (see §4.2.1 — Web Search Fallback):
   → if chunk count < FALLBACK_MIN_CHUNKS OR top reranker score < FALLBACK_SCORE_THRESHOLD:
       call Tavily API → merge web results with any existing Qdrant chunks
5. If combined context is empty → return "Xin lỗi, tôi không thể tìm thấy thông tin liên quan." without calling LLM
6. Build prompt from template (see §4.2.2)
7. Generate answer via LLM (full response, not streaming)
8. Guard check (output):
   → safe: proceed
   → unsafe: retry once with safety feedback (see §4.2.3) appended to prompt
   → still unsafe: return apology
9. Simulate streaming: split final answer on whitespace, yield each word as an SSE data chunk with ~15ms inter-chunk delay. Citations are appended as a single final chunk.
```

#### 4.2.1 Web Search Fallback

**Trigger conditions (either is sufficient):**

| Condition | Config var | Default |
|-----------|-----------|---------|
| Reranked chunk count < threshold | `FALLBACK_MIN_CHUNKS` | `3` |
| Top reranker score < threshold | `FALLBACK_SCORE_THRESHOLD` | `0.5` |

**Evaluation:** After reranking, check two conditions:
1. Count how many chunks were returned by the reranker (up to `RERANK_TOP_N`). If this count is less than `FALLBACK_MIN_CHUNKS`, trigger fallback.
2. Check the highest reranker score among returned chunks. If it is below `FALLBACK_SCORE_THRESHOLD`, trigger fallback.

**Fallback behavior:**
- Call `TavilyClient.search(query=user_query, max_results=TAVILY_MAX_RESULTS)` via `tavily-python` library
- `TAVILY_MAX_RESULTS` defaults to `5`
- Merge results: Qdrant chunks listed first, Tavily results appended after
- Tavily results are tagged `source_type: "web"` internally so the prompt builder and citation block can distinguish them

**Graceful degradation:** If Tavily is unavailable or returns an error, log a warning and proceed with whatever Qdrant chunks exist — do not abort the request.

**Integration:** Tavily calls are made directly inside `pipeline/rag.py` using the `tavily-python` SDK. No new service container is needed — `TAVILY_API_KEY` is injected via env var into the orchestrator container.

#### 4.2.2 LLM System Prompt and RAG Prompt Template

**System prompt:**

```
Bạn là trợ lý chuyên gia về kinh tế và tài chính Việt Nam. Nhiệm vụ của bạn là trả lời câu hỏi của người dùng dựa trên các bài báo được cung cấp bên dưới.

Quy tắc:
- Chỉ trả lời bằng tiếng Việt.
- Chỉ sử dụng thông tin từ các bài báo được cung cấp. Không sử dụng kiến thức bên ngoài.
- Nếu các bài báo không chứa đủ thông tin để trả lời, hãy nói rõ rằng bạn không tìm thấy thông tin liên quan.
- Trích dẫn nguồn bằng số thứ tự [1], [2], ... tương ứng với các bài báo bên dưới.
- Trả lời ngắn gọn, chính xác, đi thẳng vào vấn đề.
```

**User prompt template:**

```
Ngữ cảnh từ cơ sở dữ liệu:

[1] {title_1} — {source_1}, {published_date_1}
{chunk_text_1}

[2] {title_2} — {source_2}, {published_date_2}
{chunk_text_2}

...

---
Câu hỏi: {user_query}
```

Chunks are ordered by reranker score (highest first). Each chunk includes title, source, date, and the chunk text.

**Citation block** (appended to LLM answer by the orchestrator, not generated by the LLM):

```
---
Nguồn tham khảo:
[1] {title_1} — {source_1}, {published_date_1} — {url_1}
[2] {title_2} — {source_2}, {published_date_2} — {url_2}
```

#### 4.2.2 Safety Retry Feedback

When the output guard flags a generated answer as unsafe, the orchestrator appends this feedback to the prompt and retries generation once:

```
Phản hồi từ hệ thống kiểm duyệt: Câu trả lời trước đó bị đánh giá là không an toàn. Vui lòng trả lời lại một cách trung lập, chỉ dựa trên thông tin trong ngữ cảnh được cung cấp. Không đưa ra ý kiến cá nhân hay nội dung gây tranh cãi. 
```

### 4.3 Open WebUI Integration

Open WebUI connects to FastAPI as an OpenAI-compatible backend via `POST /v1/chat/completions` with `stream: true`. It acts purely as a frontend shell — no plugins, no custom logic.

**Open WebUI env vars (in docker-compose.yml):**

```
OPENAI_API_BASE_URL=http://orchestrator:8000/v1
OPENAI_API_KEY=dummy                    # required by WebUI but not validated by orchestrator
ENABLE_RAG_WEB_SEARCH=false             # disable built-in RAG — ours is in FastAPI
ENABLE_IMAGE_GENERATION=false
WEBUI_AUTH=false                        # no login for Phase 1 demo
DEFAULT_MODELS=multimodal-rag           # the model name orchestrator advertises
```

### 4.4 Configuration (env vars)

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
- **Not** a classical cross-encoder. Uses yes/no token logits to compute relevance.
- **Hardcoded instruction:** `"Cho mot cau hoi ve kinh te tai chinh, danh gia muc do lien quan cua doan van ban voi cau hoi."`

**Critical:** Input must be wrapped in the model's chat template with a system judge prompt and thinking-suppression suffix. The full input preparation:

```python
from transformers import AutoModelForCausalLM, AutoTokenizer

tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen3-Reranker-0.6B", padding_side='left')
model = AutoModelForCausalLM.from_pretrained("Qwen/Qwen3-Reranker-0.6B").eval()

token_true_id = tokenizer.convert_tokens_to_ids("yes")
token_false_id = tokenizer.convert_tokens_to_ids("no")

# Chat template wrapping — required for correct logits
PREFIX = '<|im_start|>system\nJudge whether the Document meets the requirements based on the Query and the Instruct provided. Note that the answer can only be "yes" or "no".<|im_end|>\n<|im_start|>user\n'
SUFFIX = '<|im_end|>\n<|im_start|>assistant\n<think>\n\n</think>\n\n'
INSTRUCTION = "Cho mot cau hoi ve kinh te tai chinh, danh gia muc do lien quan cua doan van ban voi cau hoi."

def format_pair(query: str, document: str) -> str:
    body = f"<Instruct>: {INSTRUCTION}\n<Query>: {query}\n<Document>: {document}"
    return PREFIX + body + SUFFIX

# For each (query, document) pair:
formatted = format_pair(query, document)
inputs = tokenizer(formatted, return_tensors="pt", padding=True).to(model.device)
with torch.no_grad():
    logits = model(**inputs).logits[:, -1, :]
    true_score = logits[:, token_true_id].exp().item()
    false_score = logits[:, token_false_id].exp().item()
    relevance_score = true_score / (true_score + false_score)
```

Without the `PREFIX`/`SUFFIX` wrapping (especially the `<think>\n\n</think>\n\n` to suppress thinking mode), the model produces incorrect logits.

### 5.3 Guard Service

- **Model:** `Qwen/Qwen3Guard-Gen-0.6B`
- **Library:** `transformers` with `AutoModelForCausalLM`
- **API:**
  - Input moderation: `POST /classify` → `{ "text": "user query", "role": "input" }` → `{ "label": "safe"|"unsafe" }`
  - Output moderation: `POST /classify` → `{ "text": "llm answer", "role": "output", "prompt": "original user query" }` → `{ "label": "safe"|"unsafe" }`

**Input moderation:** Format as `[{"role": "user", "content": text}]`, apply chat template, generate up to 128 tokens.

**Output moderation:** Format as `[{"role": "user", "content": prompt}, {"role": "assistant", "content": text}]`, apply chat template, generate up to 128 tokens. The `prompt` field provides the original user query as context for judging the response.

Parse output: look for `Safety: Safe` / `Safety: Unsafe` / `Safety: Controversial`. Anything non-`Safe` is treated as unsafe.

### 5.4 LLM Service (Remote)

- **Model:** `Qwen/Qwen3.5-4B`
- **Serving:** `vllm serve Qwen/Qwen3.5-4B --port 8004 --reasoning-parser qwen3`
- Orchestrator calls vLLM's native `/v1/chat/completions` via OpenAI SDK.
- **Thinking mode disabled** for RAG answers: `extra_body={"chat_template_kwargs": {"enable_thinking": False}}`
- System prompt enforces Vietnamese-only answers with citation format.

---

> **Below this line: future phase references — NOT in Phase 1 scope.**

### 5.5 ASR Service (Phase 2)

- **Model:** `Qwen/Qwen3-ASR-1.7B` via `qwen-asr` package. Port 8005, local GPU.
- Full implementation details deferred to Phase 2 spec.

### 5.6 VLM Service (Phase 3)

- **Model:** `Qwen/Qwen3-VL-4B-Instruct` via `transformers`. Port 8006, remote GPU.
- Full implementation details deferred to Phase 3 spec.

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

**Health checks:** Every model service exposes `GET /health` that returns HTTP 200 only after weights are loaded and the model is ready to serve. Docker Compose `healthcheck` directives use this with `interval: 10s`, `start_period: 120s` (GPU models take 30–120s to load). The `orchestrator` service uses `depends_on` with `condition: service_healthy` for `embedding`, `reranker`, `guard`, and `qdrant`.

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
| Qdrant returns 0 results | Return "no relevant context found" message without calling LLM |

The apology message (`APOLOGY_MESSAGE`) is used for guardrail failures only. HTTP 503s are distinct so WebUI shows a "service unavailable" message. All timeouts configured via env vars.

**Concurrency:** Model services handle requests sequentially (single-worker uvicorn). Concurrent orchestrator requests are handled via async I/O; GPU service calls queue naturally. For Phase 1 demo traffic levels, this is acceptable. If load increases, add gunicorn workers or model replicas.

## 8. Extensibility

### Phase 2 — Audio

Add `POST /v1/audio/transcriptions` to the orchestrator. It calls the ASR service, gets a transcript, and re-routes it through the existing RAG pipeline. The `/v1/chat/completions` endpoint stays untouched.

### Phase 3 — Image/Chart

The orchestrator detects image attachments in the OpenAI-format message (base64 in `content` array), calls the VLM service for structured text extraction, then runs the standard RAG pipeline on the resulting text query. No changes to retrieval, reranking, or LLM layers.

### Phase 4 — PDF/Docling

Add `scripts/ingest_pdf.py` using Docling to extract text. Chunk using the same semantic strategy. Upsert into the same Qdrant collection with `source_type: "pdf"` in the payload. The retrieval and generation pipeline requires zero changes.

### Design Invariant

Everything downstream of `normalize_query(text) → str` is modality-agnostic. ASR, VLM, and future Docling extraction all reduce to text before entering retrieval.
