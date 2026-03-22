# Central Config + Loguru Semantic Logging — Design Spec

**Date:** 2026-03-22
**Status:** Approved
**Scope:** Multimodal Economic RAG Assistant

---

## 1. Problem Statement

83 hardcoded values are scattered across 11+ files: model names, prompts, timeouts,
generation parameters, and magic numbers. All 11 files use Python stdlib `logging`
with no structured format or domain-specific levels. Changing any parameter requires
finding and editing application code rather than configuration.

---

## 2. Goals

1. **Single source of truth** — all tuneable parameters defined in one config module;
   override any value via `.env` at runtime without touching code.
2. **Semantic logging** — replace stdlib `logging` with loguru everywhere; add 4
   RAG-domain log levels (`RETRIEVAL`, `RERANK`, `GUARD`, `LLM`) for pipeline-stage
   filtering.
3. **Folder clarity** — rename top-level `orchestrator/` → `api/` to eliminate the
   confusing `orchestrator/orchestrator/` double-nesting.

---

## 3. Non-Goals

- No changes to prompt content (only extracting to config).
- No changes to service container Dockerfiles.
- No new external config format (no YAML/TOML).
- No changes to the Python package name (`orchestrator`).

---

## 4. Folder Rename

| Before | After |
|---|---|
| `orchestrator/` | `api/` |
| `orchestrator/orchestrator/` | `api/orchestrator/` (Python package unchanged) |
| `services/` | `services/` (unchanged) |

`docker-compose.yml` build context updated: `context: ./orchestrator` → `context: ./api`.
All Python imports (`from orchestrator.config import ...`) are unaffected.

---

## 5. Config Architecture

### 5.1 Nested Groups

The monolithic `Settings` class is replaced with **5 nested config groups** composed
under a single `Settings` root. Only `Settings` inherits from `BaseSettings` and reads
`.env`. All sub-groups are plain `BaseModel` — pure data containers, testable without
env setup.

```
Settings (BaseSettings)
├── llm:          LLMConfig
├── services:     ServicesConfig
├── rag:          RAGConfig
├── prompts:      PromptsConfig
└── observability: ObservabilityConfig
```

Pydantic `env_nested_delimiter="__"` enables `.env` overrides with `GROUP__FIELD=value`:

```env
SERVICES__EMBEDDING_MODEL=intfloat/multilingual-e5-large
RAG__RETRIEVAL_TOP_K=30
LLM__TEMPERATURE=0.5
```

### 5.2 Field Definitions

**LLMConfig**
```python
url: str = "http://localhost:8004"
model: str = "Qwen/Qwen3.5-4B"
temperature: float = 0.7
max_tokens: int = 512
timeout: float = 60.0
```

**ServicesConfig**
```python
embedding_url: str = "http://embedding:8001"
embedding_model: str = "Qwen/Qwen3-Embedding-0.6B"
embedding_timeout: float = 15.0

reranker_url: str = "http://reranker:8002"
reranker_model: str = "Qwen/Qwen3-Reranker-0.6B"
reranker_timeout: float = 15.0

guard_url: str = "http://guard:8003"
guard_model: str = "Qwen/Qwen3Guard-Gen-0.6B"
guard_timeout: float = 10.0

qdrant_url: str = "http://qdrant:6333"
qdrant_collection: str = "econ_vn_news"
```

**RAGConfig**
```python
retrieval_top_k: int = 20
rerank_top_n: int = 5
fallback_min_chunks: int = 3
fallback_score_threshold: float = 0.5
context_limit: int = 5
citation_limit: int = 5
```

**PromptsConfig**
```python
system_prompt: str = "Bạn là trợ lý AI chuyên về kinh tế tài chính Việt Nam. ..."
user_template: str = "Dựa vào các đoạn văn bản sau:\n{context}\n\nTrả lời: {question}"
reranker_instruction: str = "Cho một câu hỏi về kinh tế, tài chính, đánh giá mức độ liên quan..."
no_context_message: str = "Xin lỗi, tôi không tìm thấy thông tin liên quan."
guard_error_message: str = "Xin lỗi, yêu cầu của bạn không thể xử lý."
apology_message: str = "Xin lỗi, tôi không thể trả lời câu hỏi này theo nội dung của chúng tôi."
```

**ObservabilityConfig**
```python
log_level: str = "INFO"          # overridable via LOG_LEVEL env var
langsmith_api_key: str | None = None
langsmith_project: str = "multimodal-rag"
tavily_api_key: str | None = None
```

### 5.3 Access Pattern Changes

| Before | After |
|---|---|
| `settings.retrieval_top_k` | `settings.rag.retrieval_top_k` |
| `settings.llm_timeout` | `settings.llm.timeout` |
| `settings.embedding_url` | `settings.services.embedding_url` |
| `settings.guard_url` | `settings.services.guard_url` |
| `settings.langsmith_api_key` | `settings.observability.langsmith_api_key` |
| `settings.no_context_message` | `settings.prompts.no_context_message` |

### 5.4 Service Containers (embedding, reranker, guard)

These containers use `os.getenv()` and do not share the `orchestrator` package.
Model names are added to `.env` and passed via `docker-compose.yml` environment
blocks. No code changes to how they read env vars.

```yaml
# docker-compose.yml
embedding:
  environment:
    - EMBEDDING_MODEL=${SERVICES__EMBEDDING_MODEL:-Qwen/Qwen3-Embedding-0.6B}
```

---

## 6. Loguru Integration

### 6.1 Setup Location

`api/orchestrator/tracing.py` — the existing observability module — becomes the
single loguru configuration point. It:

1. Removes the default loguru sink
2. Adds a new stderr sink with structured format
3. Registers 4 custom domain levels
4. Intercepts stdlib `logging` records and routes them through loguru
   (covers uvicorn, httpx, qdrant-client)

### 6.2 Custom Domain Levels

| Level | Number | Color | Semantic meaning |
|---|---|---|---|
| `RETRIEVAL` | 25 | cyan | Dense/sparse vector search — hits, scores, top-k |
| `RERANK` | 26 | blue | Cross-encoder reranking — input/output count, top score |
| `GUARD` | 27 | yellow | Safety classification — role, label, latency |
| `LLM` | 28 | magenta | Generation — model, token count, latency |

All 4 levels sit between `INFO` (20) and `WARNING` (30), so `LOG_LEVEL=INFO` shows
them and `LOG_LEVEL=WARNING` suppresses them cleanly.

### 6.3 Log Format

```
{time:YYYY-MM-DD HH:mm:ss.SSS} | {level:<10} | {name}:{function}:{line} | {message}
```

Per-request context bound via `logger.bind(request_id=...)` in the chat router,
propagated through pipeline stages.

### 6.4 Example Output

```
2026-03-22 14:01:23.451 | RETRIEVAL  | retriever:retrieve:87  | req=abc123 top_k=20 dense=18 sparse=15
2026-03-22 14:01:23.891 | RERANK     | reranker:rerank:44     | req=abc123 input=18 output=5 top_score=0.94
2026-03-22 14:01:24.102 | GUARD      | guard:classify:31      | req=abc123 role=input label=safe latency_ms=210
2026-03-22 14:01:25.330 | LLM        | llm:generate:52        | req=abc123 model=Qwen/Qwen3.5-4B tokens=487 latency_ms=1228
```

### 6.5 Usage in Each File

```python
from loguru import logger

# Standard usage
logger.info("Server started")
logger.error("Connection failed: {}", err)

# Domain levels
logger.log("RETRIEVAL", "top_k={} dense={} sparse={}", top_k, dense, sparse)
logger.log("GUARD", "role={} label={} latency_ms={}", role, label, ms)
```

Service containers (embedding, reranker, guard) each get a minimal loguru setup
replacing any stdlib `logging` calls — no custom levels needed, just INFO/ERROR.

---

## 7. Files Changed

### api/ (was orchestrator/)
| File | Change |
|---|---|
| `api/orchestrator/config.py` | Full rewrite — 5 nested config groups |
| `api/orchestrator/tracing.py` | Loguru setup, custom levels, stdlib intercept |
| `api/orchestrator/main.py` | `logging` → `loguru`; use `settings.observability` |
| `api/orchestrator/pipeline/rag.py` | Prompts/limits → `settings.prompts/rag` |
| `api/orchestrator/services/llm.py` | Params → `settings.llm.*`; `logging` → `loguru` + `LLM` level |
| `api/orchestrator/services/embedder.py` | URL/timeout → `settings.services.*`; loguru |
| `api/orchestrator/services/reranker.py` | URL/timeout/instruction → settings; loguru + `RERANK` level |
| `api/orchestrator/services/retriever.py` | URL/collection/top_k → settings; loguru + `RETRIEVAL` level |
| `api/orchestrator/services/guard.py` | URL/timeout → settings; loguru + `GUARD` level |
| `api/orchestrator/services/web_search.py` | loguru |
| `api/orchestrator/routers/chat.py` | loguru; bind `request_id` |

### services/
| File | Change |
|---|---|
| `services/embedding/embedding_app.py` | loguru (minimal setup) |
| `services/reranker/reranker_app.py` | loguru; `RERANKER_MODEL` env var (already uses os.getenv) |
| `services/guard/guard_app.py` | loguru; `GUARD_MODEL` env var (already uses os.getenv) |

### Root
| File | Change |
|---|---|
| `docker-compose.yml` | Build context `./orchestrator` → `./api`; add model env vars |
| `.env` + `.env.example` | Add `SERVICES__*_MODEL`, `LOG_LEVEL` |
| `pyproject.toml` | Add `loguru>=0.7` dependency |
| `tests/orchestrator/` | Update config access patterns (e.g. `s.rag.retrieval_top_k`) |

---

## 8. Dependency Addition

```toml
# pyproject.toml
"loguru>=0.7.0",
```

```txt
# api/requirements.txt
loguru==0.7.3
```

---

## 9. Testing Strategy

- **Unit:** `tests/orchestrator/test_config.py` updated — instantiate each sub-group
  directly with dict kwargs (no env needed); assert defaults and nested access paths.
- **Integration:** Existing tests pass unchanged after access pattern updates
  (`s.retrieval_top_k` → `s.rag.retrieval_top_k`).
- **Log output:** Loguru's `capfd` / `capsys` capture works with pytest;
  no special fixtures needed.

---

## 10. Rollout Order

1. Git mv `orchestrator/` → `api/`; update `docker-compose.yml` context
2. Rewrite `config.py` with nested groups
3. Update `tracing.py` with loguru setup + custom levels
4. Update all orchestrator service/pipeline/router files
5. Update service container apps (embedding, reranker, guard)
6. Update `.env`, `.env.example`, `pyproject.toml`, `requirements.txt`
7. Update tests
8. Rebuild Docker images and verify `docker compose up -d`
