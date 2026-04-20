# Multimodal Economic RAG Assistant

> 🇬🇧 **English** | [🇻🇳 Tiếng Việt](README.vi.md)

An OpenAI-compatible Retrieval-Augmented Generation system for Vietnamese economic and financial Q&A.

The stack pairs a **FastAPI orchestrator** (LangGraph workflow) with a **consolidated inference service** (embedding + reranking on GPU), a **Qdrant** vector database, a **vLLM** LLM backend, and an **Open WebUI** frontend.

---

## Table of Contents

1. [What This Project Does](#1-what-this-project-does)
2. [Architecture Overview](#2-architecture-overview)
3. [LangGraph State Machine](#3-langgraph-state-machine)
4. [RAG Pipeline Deep Dive](#4-rag-pipeline-deep-dive)
5. [Repository Structure](#5-repository-structure)
6. [Runtime Services](#6-runtime-services)
7. [Key Concepts & Design Choices](#7-key-concepts--design-choices)
8. [Configuration Reference](#8-configuration-reference)
9. [Installation & Quickstart](#9-installation--quickstart)
10. [Development Workflow](#10-development-workflow)
11. [Testing](#11-testing)
12. [Troubleshooting](#12-troubleshooting)

---

## 1. What This Project Does

Users send economic or financial questions in Vietnamese. The system retrieves relevant passages from an internal corpus (Vietnamese economic news), optionally augments with live web search, then generates grounded answers with inline citations (`[S1]`, `[S2]` → clickable links).

**Key capabilities:**

- Hybrid dense+sparse retrieval with Reciprocal Rank Fusion (RRF)
- Smart web-search fallback triggered by low-confidence or time-sensitive queries
- Inline citation generation with post-processing to clickable markdown
- Intent routing — the LLM decides per query whether to use RAG or answer directly
- OpenAI-compatible `/v1/chat/completions` endpoint (streaming and non-streaming)

---

## 2. Architecture Overview

```text
┌────────────────────────────────────────────────────────────────────────┐
│                           User / Open WebUI                            │
│                      POST /v1/chat/completions                         │
└─────────────────────────────┬──────────────────────────────────────────┘
                              │
                    ┌─────────▼─────────┐
                    │   Orchestrator    │  :8000  FastAPI + LangGraph
                    │  (Intent Router)  │
                    └──┬──────┬─────────┘
                       │      │
          RAG path     │      │  Direct path
                       │      │
          ┌────────────▼──┐  ┌▼────────────────┐
          │  RAG Graph    │  │  Direct LLM call │
          │  (LangGraph)  │  │  (± web context) │
          └──┬──────┬─────┘  └─────────────────┘
    embed    │      │ retrieve
             │      │
    ┌────────▼──┐  ┌▼────────────┐  ┌──────────────┐
    │ Inference │  │   Qdrant    │  │ Tavily (web) │
    │  :8001    │  │  :6333      │  │  (optional)  │
    │  embed    │  │  hybrid     │  │  fallback    │
    │  sparse   │  │  search     │  └──────────────┘
    │  rerank   │  └─────────────┘
    └───────────┘
             │
    ┌────────▼──────────┐
    │  LLM (vLLM)       │
    │  :8004            │
    │  Qwen3-4B-AWQ     │
    └───────────────────┘
```

**Request flow:**

1. Orchestrator receives chat request → calls `detect_intent()` (LLM JSON routing).
2. **Direct path:** LLM answers immediately, optionally with web context.
3. **RAG path:** LangGraph runs embed → retrieve → rerank → web fallback → combine → generate → citations.
4. Response is streamed or returned as a single completion.

---

## 3. LangGraph State Machine

The entire RAG pipeline is encoded as a **LangGraph `StateGraph`** over `RAGState`.

### State Schema

```text
RAGState
├── query            str           normalized query
├── raw_query        str           original user input
├── resolved_query   str           intent-resolved query
├── task_type        str           "rag" | "direct" | "direct_web"
├── embeddings       list[float]   dense query vector
├── retrieved_docs   list[dict]    hybrid search results (top_k=20)
├── reranked_docs    list[dict]    reranked results (top_n=5)
├── web_results      list[dict]    Tavily web search results
├── final_context    list[dict]    merged reranked + web
├── answer           str           generated answer text
├── generation_prompt str          rendered prompt sent to LLM
├── citations        list[dict]    normalized citation metadata
├── citation_pool    dict          context_id → full item metadata
└── error            str | None    set on node failure
```

### Graph Topology

```text
                        START
                          │
                   ┌──────▼──────┐
                   │  embed_node  │  dense vector via InferenceClient
                   └──────┬──────┘
                          │ error? → END
                   ┌──────▼──────┐
                   │retrieve_node│  hybrid_search (dense + sparse RRF)
                   └──────┬──────┘
                   ┌──────▼──────┐
                   │ rerank_node │  bge-reranker-v2-m3, top_n results
                   └──────┬──────┘
                   ┌──────▼──────────┐
                   │web_fallback_node│  policy check → Tavily if triggered
                   └──────┬──────────┘
                   ┌──────▼──────────────┐
                   │combine_context_node │  merge docs + build citation_pool
                   └──────┬──────────────┘
                          │  retrieval_only=True → END (used for streaming)
                   ┌──────▼──────┐
                   │generate_node│  LLMClient.generate() with [S1][S2] prompt
                   └──────┬──────┘
                   ┌──────▼───────────┐
                   │  citations_node  │  rewrite [Sn] → [[Sn]](url) markdown
                   └──────┬───────────┘
                          END
```

**Streaming mode:** The graph is invoked with `retrieval_only=True` to build the context, then generation is streamed directly via `LLMClient.stream_chat()` outside the graph.

---

## 4. RAG Pipeline Deep Dive

### 4.1 Hybrid Retrieval (RRF)

Two parallel search paths are fused:

| Path | Model | Index type |
|------|-------|-----------|
| Dense | BAAI/bge-m3 (1024-dim) | cosine HNSW |
| Sparse | BAAI/bge-m3 lexical weights | Qdrant sparse |

Results are combined with **Reciprocal Rank Fusion** — each document's final score is `Σ 1 / (k + rank_i)`. This balances exact-keyword recall (sparse) with semantic recall (dense).

### 4.2 Reranking

The top-k retrieved documents (default 20) are re-scored with `BAAI/bge-reranker-v2-m3` (cross-encoder, query+passage pairs). Scores are sigmoid-normalized. Only the top-n survive (default 5).

### 4.3 Web Fallback Policy

`should_add_web_fallback()` in `rag_policy.py` triggers web search when corpus support is weak:

| Condition | Reason tag |
|-----------|-----------|
| No reranked docs | `no_docs` |
| Top score < 0.70 | `hard_below` |
| Top score < 0.85 **and** shallow support (< 2 docs, gap ≥ 0.12) | `shallow` |
| Time-sensitive markers detected ("hôm nay", "mới nhất", "current", "latest") | `time_sensitive` |
| Query significantly expanded (4+ new tokens vs original) | `expansion` |
| Top score ≥ 0.85 and sufficient support | skip — `soft_above` |
| No Tavily key | skip — `disabled` |

### 4.4 Citation System

```
LLM generates:   "GDP tăng 6.5% [S1] nhờ xuất khẩu mạnh [S2]."
                           ↓ finalize_citations()
Final answer:    "GDP tăng 6.5% [[S1]](https://...) nhờ xuất khẩu mạnh [[S2]](https://...)."
                           +
                 ### Nguồn trích dẫn
                 - [S1] Title of source 1 — score: 0.92
                 - [S2] Title of source 2 — score: 0.87
```

`rewrite_inline_citations()` is regex-based and skips fenced code blocks to avoid corrupting code examples.

---

## 5. Repository Structure

```
.
├── Makefile                         # All dev & ops commands
├── pyproject.toml                   # Python dependencies (uv)
├── .env.example                     # All config variables with defaults
├── docker-compose.yaml              # Production stack
├── docker-compose.dev.yaml          # Dev stack (bind mounts, hot reload)
│
├── api/
│   └── orchestrator/
│       ├── config.py                # Pydantic Settings (nested, env-driven)
│       ├── main.py                  # FastAPI app factory + lifespan clients
│       ├── tracing.py               # Loguru + LangSmith setup
│       ├── models/
│       │   └── schemas.py           # OpenAI-compatible message schemas
│       ├── pipeline/
│       │   ├── rag.py               # LangGraph StateGraph definition
│       │   ├── rag_policy.py        # Web fallback heuristics
│       │   ├── rag_context.py       # Context merge + citation post-processing
│       │   └── rag_prompts.py       # Prompt templates (Vietnamese)
│       ├── routers/
│       │   └── chat.py              # POST /v1/chat/completions
│       └── services/
│           ├── llm.py               # LLMClient (OpenAI async wrapper)
│           ├── inference.py         # InferenceClient (embed/sparse/rerank)
│           ├── retriever.py         # RetrieverClient (Qdrant hybrid search)
│           ├── web_search.py        # WebSearchClient (Tavily)
│           └── conversation.py      # Message normalization
│
├── services/
│   └── inference/
│       ├── inference_app.py         # FastAPI: /embed /sparse /rerank /health
│       └── requirements.txt
│
├── scripts/
│   ├── ingest.py                    # HuggingFace dataset → Qdrant
│   ├── chunker.py                   # Semantic chunking
│   ├── qdrant_bootstrap.py          # Create collection + vector indexes
│   └── qdrant_snapshot_restore.py   # Restore from .snapshot file
│
├── infra/docker/
│   ├── app.cpu.Dockerfile           # Orchestrator + ingest (CPU)
│   └── app.gpu.Dockerfile           # Inference service (GPU, CUDA 12.8)
│
├── data/
│   └── *.snapshot                   # Qdrant snapshot for reproducible data
│
└── tests/
    └── orchestrator/                # Unit + integration tests
```

---

## 6. Runtime Services

| Service | Port | GPU | Description |
|---------|------|-----|-------------|
| `orchestrator` | 8000 | — | FastAPI + LangGraph pipeline |
| `inference` | 8001 | GPU 0 | Embedding + sparse encoding + reranking |
| `llm` |  | GPU 1 | vLLM OpenAI-compatible endpoint (Rented in Vast.ai) |
| `qdrant` | 6333 / 6334 | — | Vector database (REST / gRPC) |
| `webui` | 8080 | — | Open WebUI chat interface |
| `bootstrap` | — | — | One-off: create Qdrant collection (profile: `tools`) |
| `ingest` | — | — | One-off: load data into Qdrant (profile: `ingest`) |
| `tunnel` | — | — | Cloudflare tunnel (optional) |

The `llm` service is commented out by default — point `LLM__URL` to an external vLLM instance instead.

---

## 7. Key Concepts & Design Choices

### 7.1 Intent Routing (Direct vs. RAG)

**Concept:** Not every question benefits from RAG. Some are general knowledge ("What is GDP?") or conversational ("Hi there!"). The system uses an LLM to classify each query:

- **Direct:** Answer immediately from LLM knowledge
- **Direct + Web:** Direct answer + augment with live web results
- **RAG:** Full retrieval pipeline for domain-specific questions

**Design choice:** This saves retrieval+reranking latency for ~30% of queries while improving answer freshness for time-sensitive queries. The classifier runs inline — no extra service.

### 7.2 Hybrid Retrieval (Dense + Sparse + RRF)

**Concept:** Two complementary search modes:

- **Dense embeddings** (BAAI/bge-m3): Capture semantic meaning. "GDP growth" matches "economic expansion" via 1024-dim similarity.
- **Sparse lexical** (BM25 weights): Exact keyword matching. "GDP growth" matches "GDP growth" word-for-word.

Neither alone is enough. A query about "Thị trường chứng khoán Việt Nam năm 2026" (Vietnamese stock market 2026) needs both semantic match ("market sentiment") and exact terms ("2026").

**Reciprocal Rank Fusion (RRF):** Combines both rankings via `Σ 1 / (k + rank_i)` — parameter-free, robust. A document ranked #1 in dense and #10 in sparse gets a strong combined score.

**Design choice:** Hybrid search avoids parameter tuning (no alpha blend factor) and provides insurance — if one mode fails, the other partially compensates.

### 7.3 Web Fallback Policy (Adaptive Augmentation)

**Concept:** The internal corpus is sparse (~50k economic articles). When retrieval confidence is low, web search augments:

| Signal | Meaning |
| --- | --- |
| `hard_below` (score < 0.70) | Weak match → web search compensates |
| `shallow` (score < 0.85 + thin support) | Marginal match with only 1–2 sources → web provides diversity |
| `time_sensitive` ("hôm nay", "2026") | Recency matters → grab fresh web results |
| `expansion` (4+ new tokens) | Query changed significantly → corpus may be outdated |

**Design choice:** Fallback is policy-driven, not threshold-driven. Instead of hard cutoff at 0.75, we ask "is this good enough?" — considering not just score but context depth, recency, and expansion. This reduces false positives (web spam) while catching real gaps.

### 7.4 Reranking with Cross-Encoders

**Concept:** Dense retrieval returns top-20 candidates fast. But ranking by similarity is coarse — all 20 docs might be "similar." Reranking uses a cross-encoder (`BAAI/bge-reranker-v2-m3`), which jointly encodes query+passage (not separately like dense). This refines the top-5:

```text
Before rerank:  [News1:0.82, News2:0.81, News3:0.79, News4:0.78, News5:0.77, Spam:0.76]
After rerank:   [News1:0.92, News3:0.87, News2:0.71, Spam:0.45]  ← precise ranking
```

**Design choice:** Cross-encoders are slower (few ms per pair) but worth it for the ~8 docs passed to the LLM. Dense-only ranking can flip top results unexpectedly; reranking stabilizes final context.

### 7.5 Citation System with Post-Processing

**Concept:** LLM generates answers with inline markers `[S1]`, `[S2]`. These are heuristic (not guaranteed to match retrieved sources). Post-processing:

1. Extracts `[Sn]` markers → checks citation_pool for metadata
2. Rewrites to clickable markdown `[[S1]](url)`
3. Appends a "Sources" section with titles, URLs, and confidence scores

**Design choice:** Inline citations are more transparent than footnotes. They show which sentence relies on which source. Post-processing ensures all citations are valid (404 checks skipped for speed, but URLs are verified at ingest).

### 7.6 LangGraph over Direct Python

**Concept:** LangGraph is a state machine framework. Instead of:

```python
# ❌ Monolithic function
def rag_pipeline(query):
    embed = embed_query(query)
    docs = retrieve(embed)
    reranked = rerank(docs)
    web = fallback_policy(reranked) and web_search(query)
    context = combine(reranked, web)
    answer = generate(context)
    return finalize_citations(answer, context)
```

We use:

```python
# ✅ Composable nodes
graph = StateGraph(RAGState)
graph.add_node("embed", embed_node)
graph.add_node("retrieve", retrieve_node)
graph.add_conditional_edge("retrieve", fallback_policy_router)
graph.compile()
```

**Design choice:** This gives us:

- **Inspectability:** See the graph visually in LangSmith
- **Testability:** Mock individual nodes
- **Streaming:** Separate retrieval from generation (retrieval_only mode)
- **Observability:** Each node is automatically traced

### 7.7 Dedicated Inference Service

**Concept:** Embedding + reranking are GPU-bound. Orchestrator is CPU-bound. Splitting them:

```text
Orchestrator (CPU)  ←→  Inference (GPU 0)  +  LLM (GPU 1)
      ↑                         ↑
  Routes requests          Embedding, sparse, rerank
  Coordinates flow         (reusable across replicas)
```

**Design choice:** One inference service can serve 10 orchestrator replicas. This amortizes GPU costs and allows independent scaling. If embedding is slow, add more GPUs to inference without touching orchestrator.

### 7.8 AWQ Quantization for Qwen

**Concept:** Qwen-4B in FP16 is ~8 GB. AWQ quantization reduces it to ~3 GB with minimal quality loss (perplexity drop < 0.5%). This fits comfortably on a single consumer GPU.

**Design choice:** Quantization trades latency (2–5% slower) for VRAM. Given 4B model is already fast (< 100ms per token), the tradeoff is worthwhile for cost and availability.

### 7.9 OpenAI Compatibility

**Concept:** The `/v1/chat/completions` endpoint follows OpenAI's spec:

```json
{
  "model": "Economic-RAG-Assistant",
  "messages": [{"role": "user", "content": "What's the current exchange rate?"}],
  "stream": false
}
```

Any client using `openai-python`, `curl`, or Open WebUI works without changes.

**Design choice:** Compatibility reduces lock-in and allows easy swapping with other LLMs. Clients aren't tied to this project.

---

## 8. Configuration Reference

All variables use the nested separator `__` (e.g., `LLM__MODEL` maps to `settings.llm.model`).

```env
# ── LLM ──────────────────────────────────────────────────────────────────
LLM__URL=http://llm:8004/v1
LLM__MODEL=quannguyen204/Qwen3-4B-Instruct-2507-AWQ-W4A16
LLM__TEMPERATURE=0.7
LLM__MAX_TOKENS=1024
LLM__TIMEOUT=60.0
LLM__API_KEY=dummy

# ── vLLM runtime ──────────────────────────────────────────────────────────
VLLM__MAX_MODEL_LEN=16384
VLLM__GPU_MEMORY_UTILIZATION=0.95
VLLM__QUANTIZATION=compressed-tensors

# ── Services ──────────────────────────────────────────────────────────────
SERVICES__INFERENCE_URL=http://inference:8001
SERVICES__QDRANT_URL=http://qdrant:6333
SERVICES__QDRANT_COLLECTION=academic_chunks

# ── RAG tuning ────────────────────────────────────────────────────────────
RAG__RETRIEVAL_TOP_K=20
RAG__RERANK_TOP_N=5
RAG__WEB_FALLBACK_HARD_THRESHOLD=0.70
RAG__WEB_FALLBACK_SOFT_THRESHOLD=0.85
RAG__CONTEXT_LIMIT=5
RAG__CITATION_LIMIT=5

# ── Ingestion ─────────────────────────────────────────────────────────────
INGEST__FORCE_RECREATE=false
SNAPSHOT_HF_REPO=quannguyen204/economic-rag-snapshots
SNAPSHOT_FILENAME=academic_chunks-*.snapshot

# ── Observability ─────────────────────────────────────────────────────────
OBSERVABILITY__LOG_LEVEL=INFO
OBSERVABILITY__LANGSMITH_API_KEY=
OBSERVABILITY__TAVILY_API_KEY=       # Web search fallback (optional)

# ── Auth & infra ──────────────────────────────────────────────────────────
HF_TOKEN=                            # Required for model downloads
DOCKERHUB_NAMESPACE=your-namespace
WEBUI_SECRET_KEY=admin
CLOUDFLARE_TUNNEL_TOKEN=             # Optional
```

---

## 9. Installation & Quickstart

### Prerequisites

- Docker with the `docker compose` plugin
- NVIDIA Container Toolkit (for GPU services)
- `uv` — Python package manager

```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
```

### Option A — Full stack from snapshot (recommended)

```bash
# 1. Clone and configure
git clone https://github.com/minhquana1906/Multimodal-Economic-RAG-Assistant.git
cd Multimodal-Economic-RAG-Assistant
make setup                  # copy .env.example → .env
# Edit .env: set HF_TOKEN and optionally OBSERVABILITY__TAVILY_API_KEY

# 2. Start everything (pulls images, restores snapshot, starts services)
make start

# 3. Verify
curl http://localhost:8000/health
curl http://localhost:6333/collections
# Open WebUI at http://localhost:8080
```

### Option B — Dev stack (hot reload)

```bash
make setup                  # copy .env.example → .env
# Edit .env
make dev                    # builds images from source with bind mounts
# Edit files under api/ or services/inference/ — restart container to pick up changes
make dev-logs               # tail all logs
make dev-stop
```

### Common Make Targets

| Target | Description |
|--------|-------------|
| `make start` | Full stack: pull → snapshot restore → up |
| `make stop` | Stop all containers |
| `make dev` | Dev stack with bind mounts |
| `make dev-stop` | Stop dev stack |
| `make logs [SERVICE]` | Follow logs |
| `make ps` | Show container status |
| `make test` | Run unit tests |
| `make test-integration` | Run integration tests |
| `make bootstrap` | Create Qdrant collection locally |
| `make snapshot-restore` | Restore Qdrant from .snapshot |
| `make build` | Build Docker images locally |
| `make push` | Push images to Docker Hub |

---

## 10. Development Workflow

```bash
# Local Python environment
uv sync --dev

# Run tests
uv run pytest                         # unit tests
uv run pytest -m integration          # integration tests only

# Dev compose (hot reload via bind mounts)
make dev
make dev-logs orchestrator            # tail orchestrator only
make restart orchestrator             # pick up code changes
```

Logs use domain-specific levels (`RETRIEVAL`, `RERANK`, `LLM`) for easy filtering.

LangSmith tracing is optional — set `OBSERVABILITY__LANGSMITH_API_KEY` to enable end-to-end trace visualization.

---

## 11. Testing

The test suite covers:

| Test file | Coverage |
|-----------|----------|
| `test_rag_pipeline.py` | Graph nodes, state transitions |
| `test_rag_helpers.py` | Policy decisions, citation rewriting |
| `test_chat_flow.py` | End-to-end chat routing |
| `test_chat_router.py` | HTTP endpoint behavior |
| `test_llm.py` | LLMClient methods |
| `test_retriever.py` | Hybrid search, fallback to dense |
| `test_inline_citations.py` | Citation post-processing |
| `test_regression_matrix.py` | Factual, follow-up, sparse, no-context, web, streaming scenarios |

```bash
uv run pytest                         # all unit tests
uv run pytest tests/orchestrator/test_regression_matrix.py -m integration
```

---

## 12. Troubleshooting

### Services never become healthy

- Check `HF_TOKEN` is set and valid (required for model downloads).
- Verify NVIDIA Container Toolkit is installed: `nvidia-smi` inside container.
- Check GPU VRAM — inference needs ~4 GB, LLM needs ~6–8 GB.

### Answers fail / LLM errors

- Verify `LLM__URL` and `LLM__MODEL` match your running vLLM instance.
- Check `make logs llm` for vLLM startup errors.

### No retrieval results

- Confirm Qdrant collection exists: `curl http://localhost:6333/collections`.
- Run `make bootstrap` then `make ingest` (or `make snapshot-restore`).

### Re-ingestion fails / collection conflicts

- Set `INGEST__FORCE_RECREATE=true` to drop and recreate the collection (destructive).

### Web search not triggering

- Set `OBSERVABILITY__TAVILY_API_KEY` in `.env`. Without it, web fallback is silently disabled.

---

> 🇬🇧 **English** | [🇻🇳 Tiếng Việt](README.vi.md)
