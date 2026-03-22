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

**Test path fix required:** `tests/orchestrator/conftest.py` currently hardcodes:
```python
_orch_ctx = str(_root / "orchestrator")
sys.path.insert(0, _orch_ctx)
```
After rename this must change to:
```python
_orch_ctx = str(_root / "api")
sys.path.insert(0, _orch_ctx)
```

---

## 5. Config Architecture

### 5.1 Nested Groups

The monolithic `Settings` class is replaced with **5 nested config groups** composed
under a single `Settings` root. Only `Settings` inherits from `BaseSettings` and reads
`.env`. All sub-groups are plain `BaseModel` — pure data containers, testable without
env setup.

```
Settings (BaseSettings)
├── llm:           LLMConfig
├── services:      ServicesConfig
├── rag:           RAGConfig
├── prompts:       PromptsConfig
└── observability: ObservabilityConfig
```

`SettingsConfigDict` uses `env_nested_delimiter="__"` so any nested field is
overridable from `.env` with `GROUP__FIELD=value`:

```python
model_config = SettingsConfigDict(
    env_file=".env",
    env_file_encoding="utf-8",
    env_nested_delimiter="__",
    extra="ignore",
)
```

Example overrides:
```env
SERVICES__EMBEDDING_MODEL=intfloat/multilingual-e5-large
RAG__RETRIEVAL_TOP_K=30
LLM__TEMPERATURE=0.5
OBSERVABILITY__LOG_LEVEL=DEBUG
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

`LLMClient` is updated to accept `model`, `temperature`, and `max_tokens` in its
constructor and pass them into every `chat.completions.create()` call. `api_key`
remains a hardcoded internal constant (`"fake"`) appropriate for a local vLLM
endpoint — it is not added to config:

```python
class LLMClient:
    def __init__(self, url: str, model: str, temperature: float, max_tokens: int, timeout: float):
        self._model = model
        self._temperature = temperature
        self._max_tokens = max_tokens
        self._client = AsyncOpenAI(base_url=url, api_key="fake", timeout=timeout)
```

Constructed in `main.py` as:
```python
llm=LLMClient(
    url=settings.llm.url,
    model=settings.llm.model,
    temperature=settings.llm.temperature,
    max_tokens=settings.llm.max_tokens,
    timeout=settings.llm.timeout,
)
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

`embedding_model`, `reranker_model`, and `guard_model` in `ServicesConfig` serve two
purposes:
1. Passed into service containers via `docker-compose.yml` env vars (see §5.4).
2. Logged by the orchestrator at startup for observability (e.g., "Loaded reranker:
   Qwen/Qwen3-Reranker-0.6B"). They are **not** used by orchestrator business logic.

**RAGConfig**
```python
retrieval_top_k: int = 20
rerank_top_n: int = 5
fallback_min_chunks: int = 3
fallback_score_threshold: float = 0.5
context_limit: int = 5    # replaces hardcoded [:5] slice in rag.py context assembly
citation_limit: int = 5   # replaces hardcoded [:5] slice in rag.py citation list
```

**PromptsConfig**
```python
system_prompt: str = "Bạn là trợ lý AI chuyên về kinh tế tài chính Việt Nam. ..."
# user_template uses {context} and {question} as the only interpolation keys.
# Applied as: user_template.format(context=context_text, question=query)
user_template: str = "Dựa vào các đoạn văn bản sau:\n{context}\n\nTrả lời: {question}"
reranker_instruction: str = "Cho một câu hỏi về kinh tế, tài chính, đánh giá mức độ liên quan..."
no_context_message: str = "Xin lỗi, tôi không tìm thấy thông tin liên quan."
guard_error_message: str = "Xin lỗi, yêu cầu của bạn không thể xử lý."
apology_message: str = "Xin lỗi, tôi không thể trả lời câu hỏi này theo nội dung của chúng tôi."
```

`apology_message` is returned when `input_safe=False` in `input_guard_node` in
`rag.py`. The current code routes unsafe inputs to `__end__` leaving `answer=""`
(caller gets an empty string). After this spec, the node sets the answer explicitly:

```python
# rag.py — input_guard_node
if not safe:
    return {"input_safe": False, "answer": settings.prompts.apology_message}
```

§5.3 migration table addition: hardcoded apology string → `settings.prompts.apology_message`.

`reranker_instruction` is consumed by `RerankerClient.rerank()` in the orchestrator,
which passes it as an `instruction` field in the HTTP request payload to the reranker
service. The full wire path:

```python
# api/orchestrator/services/reranker.py — RerankerClient.rerank()
payload = {
    "query": query,
    "passages": passages,
    "instruction": instruction,   # passed from settings.prompts.reranker_instruction
}
```

```python
# services/reranker/reranker_app.py — updated RerankRequest and endpoint
class RerankRequest(BaseModel):
    query: str
    passages: list[str] = Field(..., min_length=1)
    instruction: str | None = None   # falls back to module-level INSTRUCTION if None

def format_pair(query: str, document: str, instruction: str) -> str:
    body = f"<Instruct>: {instruction}\n<Query>: {query}\n<Document>: {document}"
    return PREFIX + body + SUFFIX

@app.post("/rerank", response_model=RerankResponse)
async def rerank(request: RerankRequest):
    effective_instruction = request.instruction or INSTRUCTION
    def _score_passage(passage: str) -> float:
        formatted = format_pair(request.query, passage, effective_instruction)
        ...
```

**ObservabilityConfig**
```python
log_level: str = "INFO"          # override via OBSERVABILITY__LOG_LEVEL in .env
langsmith_api_key: str | None = None
langsmith_project: str = "multimodal-rag"
tavily_api_key: str | None = None
```

Note: the env var for log level is `OBSERVABILITY__LOG_LEVEL` (following the nested
delimiter scheme), **not** a bare `LOG_LEVEL`. The `.env.example` must document this.

### 5.3 Access Pattern Changes (complete)

| Before | After |
|---|---|
| `settings.llm_url` | `settings.llm.url` |
| `settings.llm_timeout` | `settings.llm.timeout` |
| `settings.embedding_url` | `settings.services.embedding_url` |
| `settings.embedding_timeout` | `settings.services.embedding_timeout` |
| `settings.reranker_url` | `settings.services.reranker_url` |
| `settings.reranker_timeout` | `settings.services.reranker_timeout` |
| `settings.guard_url` | `settings.services.guard_url` |
| `settings.guard_timeout` | `settings.services.guard_timeout` |
| `settings.qdrant_url` | `settings.services.qdrant_url` |
| `settings.qdrant_collection` | `settings.services.qdrant_collection` |
| `settings.retrieval_top_k` | `settings.rag.retrieval_top_k` |
| `settings.rerank_top_n` | `settings.rag.rerank_top_n` |
| `settings.fallback_min_chunks` | `settings.rag.fallback_min_chunks` |
| `settings.fallback_score_threshold` | `settings.rag.fallback_score_threshold` |
| hardcoded `[:5]` (context) | `settings.rag.context_limit` |
| hardcoded `[:5]` (citations) | `settings.rag.citation_limit` |
| `settings.no_context_message` | `settings.prompts.no_context_message` |
| `settings.guard_error_message` | `settings.prompts.guard_error_message` |
| hardcoded apology string in `rag.py` | `settings.prompts.apology_message` |
| hardcoded instruction string in `reranker_app.py` | `settings.prompts.reranker_instruction` |
| `settings.langsmith_api_key` | `settings.observability.langsmith_api_key` |
| `settings.langsmith_project` | `settings.observability.langsmith_project` |
| `settings.tavily_api_key` | `settings.observability.tavily_api_key` |

### 5.4 Service Containers (embedding, reranker, guard)

These containers use `os.getenv()` and do not share the `orchestrator` package.
Model names are passed from `.env` via `docker-compose.yml` environment blocks:

```yaml
# docker-compose.yml
embedding:
  environment:
    - EMBEDDING_MODEL=${SERVICES__EMBEDDING_MODEL:-Qwen/Qwen3-Embedding-0.6B}
reranker:
  environment:
    - RERANKER_MODEL=${SERVICES__RERANKER_MODEL:-Qwen/Qwen3-Reranker-0.6B}
guard:
  environment:
    - GUARD_MODEL=${SERVICES__GUARD_MODEL:-Qwen/Qwen3Guard-Gen-0.6B}
```

No changes to how service containers read env vars — `os.getenv("EMBEDDING_MODEL")`
etc. remain unchanged.

### 5.5 `.env` / `.env.example` Migration

All flat vars that are now covered by nested config are **removed** from `.env.example`
to avoid misleading users. The following stale flat vars are retired:

```
# REMOVED (now nested):
RETRIEVAL_TOP_K, RERANK_TOP_N, GUARD_TIMEOUT_S, LLM_TIMEOUT_S,
EMBEDDING_TIMEOUT_S, RERANKER_TIMEOUT_S, APOLOGY_MESSAGE,
RERANKER_URL, GUARD_URL, EMBEDDING_URL, QDRANT_URL
```

**Note on `OPENAI_API_BASE_URL` and `OPENAI_API_KEY`:** these are **not** orchestrator
LLM-backend vars — they configure the Open WebUI service's connection to the orchestrator
(`http://orchestrator:8000/v1`). They remain in `docker-compose.yml` under the `webui`
service with default fallbacks and are removed from `.env.example` since the defaults
are sufficient for standard deployments. Deployers who need to override the WebUI
endpoint can still set them in `.env`; the `extra="ignore"` policy means they won't
cause validation errors.

The new `.env.example` groups vars by section with comments:
```env
# === LLM Backend ===
LLM__URL=http://localhost:8004
LLM__MODEL=Qwen/Qwen3.5-4B

# === Service URLs (override if not using default docker-compose hostnames) ===
SERVICES__EMBEDDING_URL=http://embedding:8001
SERVICES__RERANKER_URL=http://reranker:8002
SERVICES__GUARD_URL=http://guard:8003
SERVICES__QDRANT_URL=http://qdrant:6333

# === Service Models ===
SERVICES__EMBEDDING_MODEL=Qwen/Qwen3-Embedding-0.6B
SERVICES__RERANKER_MODEL=Qwen/Qwen3-Reranker-0.6B
SERVICES__GUARD_MODEL=Qwen/Qwen3Guard-Gen-0.6B

# === Observability ===
OBSERVABILITY__LOG_LEVEL=INFO
OBSERVABILITY__LANGSMITH_API_KEY=your-key-here
OBSERVABILITY__LANGSMITH_PROJECT=multimodal-rag
OBSERVABILITY__TAVILY_API_KEY=your-key-here
```

---

## 6. Loguru Integration

### 6.1 Setup — `tracing.py` rewrite

`api/orchestrator/tracing.py` exports two functions:

```python
def setup_logging(config: ObservabilityConfig) -> None:
    """Configure loguru: remove default sink, add structured stderr sink,
    register custom domain levels, intercept stdlib logging."""

def setup_langsmith(config: ObservabilityConfig) -> None:
    """Enable LangSmith tracing if api_key is set."""
    if not config.langsmith_api_key:
        return
    os.environ["LANGCHAIN_TRACING_V2"] = "true"
    os.environ["LANGCHAIN_API_KEY"] = config.langsmith_api_key
    os.environ["LANGCHAIN_PROJECT"] = config.langsmith_project
    logger.info("LangSmith tracing enabled for project: {}", config.langsmith_project)
```

`main.py` calls both at startup:
```python
from orchestrator.tracing import setup_logging, setup_langsmith

setup_logging(settings.observability)
setup_langsmith(settings.observability)
```

`setup_logging` full implementation:
```python
def setup_logging(config: ObservabilityConfig) -> None:
    logger.remove()  # remove default sink

    # Default extra so {extra[request_id]} never raises KeyError outside a
    # contextualize() block (e.g. startup messages, background tasks)
    logger.configure(extra={"request_id": "-"})

    logger.add(sys.stderr, format=LOG_FORMAT, level=config.log_level, colorize=True)

    # Register custom domain levels (idempotent — skip if already registered)
    for name, no, color in DOMAIN_LEVELS:
        try:
            logger.level(name, no=no, color=color)
        except TypeError:
            pass  # already registered

    # Intercept stdlib logging → loguru (covers uvicorn, httpx, qdrant-client)
    logging.basicConfig(handlers=[_InterceptHandler()], level=0, force=True)


class _InterceptHandler(logging.Handler):
    """Route stdlib log records through loguru."""
    def emit(self, record: logging.LogRecord) -> None:
        try:
            level = logger.level(record.levelname).name
        except ValueError:
            level = record.levelno
        frame, depth = sys._getframe(6), 6
        while frame.f_code.co_filename == logging.__file__:
            frame = frame.f_back
            depth += 1
        logger.opt(depth=depth, exception=record.exc_info).log(level, record.getMessage())
```

### 6.2 Custom Domain Levels

```python
DOMAIN_LEVELS = [
    ("RETRIEVAL", 25, "<cyan>"),
    ("RERANK",    26, "<blue>"),
    ("GUARD",     27, "<yellow>"),
    ("LLM",       28, "<magenta>"),
]
```

| Level | Number | Color | Semantic meaning |
|---|---|---|---|
| `RETRIEVAL` | 25 | `<cyan>` | Dense/sparse vector search — hits, scores, top-k |
| `RERANK` | 26 | `<blue>` | Cross-encoder reranking — input/output count, top score |
| `GUARD` | 27 | `<yellow>` | Safety classification — role, label, latency |
| `LLM` | 28 | `<magenta>` | Generation — model, token count, latency |

All 4 levels sit between `INFO` (20) and `WARNING` (30):
- `OBSERVABILITY__LOG_LEVEL=INFO` → all domain levels visible
- `OBSERVABILITY__LOG_LEVEL=WARNING` → domain levels suppressed

### 6.3 Log Format

```python
LOG_FORMAT = (
    "{time:YYYY-MM-DD HH:mm:ss.SSS} | {level:<10} | "
    "{name}:{function}:{line} | {extra[request_id]} | {message}"
)
```

`{extra[request_id]}` is safe because `logger.configure(extra={"request_id": "-"})`
initialises a default before any sink is added (see §6.1). Log lines outside a
`contextualize()` block show `request_id=-`.

### 6.4 Per-Request Context (`request_id` propagation)

`chat.py` generates a short `request_id` and uses `logger.contextualize()` — a
`contextvars`-based mechanism — so all downstream calls within the same async task
automatically inherit it without being passed explicitly:

```python
# routers/chat.py
import uuid

request_id = str(uuid.uuid4())[:8]
with logger.contextualize(request_id=request_id):
    result = await rag_graph.ainvoke(state)
```

All module-level `from loguru import logger` instances automatically pick up the
contextvar value in `{extra[request_id]}` for any log call inside the `with` block.

### 6.5 Example Output

```
2026-03-22 14:01:23.451 | RETRIEVAL  | retriever:retrieve:87  | a3f9c1b2 | top_k=20 dense=18 sparse=15
2026-03-22 14:01:23.891 | RERANK     | reranker:rerank:44     | a3f9c1b2 | input=18 output=5 top_score=0.94
2026-03-22 14:01:24.102 | GUARD      | guard:classify:31      | a3f9c1b2 | role=input label=safe latency_ms=210
2026-03-22 14:01:25.330 | LLM        | llm:generate:52        | a3f9c1b2 | model=Qwen/Qwen3.5-4B tokens=487 latency_ms=1228
2026-03-22 14:01:25.335 | INFO       | main:startup:31        | -        | RAG graph ready
```

### 6.6 Usage Pattern

```python
from loguru import logger

# Standard levels
logger.info("Server started")
logger.error("Connection failed: {}", err)

# Domain levels
logger.log("RETRIEVAL", "top_k={} dense={} sparse={}", top_k, dense, sparse)
logger.log("GUARD", "role={} label={} latency_ms={}", role, label, ms)
```

### 6.7 Service Containers (embedding, reranker, guard)

Each service app gets a minimal loguru setup at module top replacing stdlib calls.
No custom levels — INFO/ERROR only. `loguru` added to each service `requirements.txt`.

```python
# top of embedding_app.py, reranker_app.py, guard_app.py
import sys
from loguru import logger
logger.remove()
logger.add(sys.stderr, format="{time:HH:mm:ss} | {level} | {message}", level="INFO")
```

---

## 7. Files Changed

### api/ (was orchestrator/)
| File | Change |
|---|---|
| `api/orchestrator/config.py` | Full rewrite — 5 nested groups + complete `SettingsConfigDict` |
| `api/orchestrator/tracing.py` | Exports `setup_logging(ObservabilityConfig)` + `setup_langsmith(ObservabilityConfig)`; registers domain levels; stdlib intercept |
| `api/orchestrator/main.py` | Call `setup_logging` + `setup_langsmith`; update `LLMClient` construction |
| `api/orchestrator/pipeline/rag.py` | Prompts → `settings.prompts.*`; `[:5]` → `settings.rag.context_limit/citation_limit`; `user_template.format(context=..., question=...)` |
| `api/orchestrator/services/llm.py` | Constructor takes `model`, `temperature`, `max_tokens`; `logging` → loguru + `LLM` level |
| `api/orchestrator/services/embedder.py` | URL/timeout from settings; `logging` → loguru |
| `api/orchestrator/services/reranker.py` | URL/timeout/instruction from settings; send `instruction` in payload; loguru + `RERANK` level |
| `api/orchestrator/services/retriever.py` | URL/collection/top_k from settings; loguru + `RETRIEVAL` level |
| `api/orchestrator/services/guard.py` | URL/timeout from settings; loguru + `GUARD` level |
| `api/orchestrator/services/web_search.py` | loguru |
| `api/orchestrator/routers/chat.py` | `logger.contextualize(request_id=...)` wrapping rag invoke |

### services/
| File | Change |
|---|---|
| `services/embedding/requirements.txt` | Add `loguru==0.7.3` |
| `services/reranker/requirements.txt` | Add `loguru==0.7.3` |
| `services/guard/requirements.txt` | Add `loguru==0.7.3` |
| `services/embedding/embedding_app.py` | Minimal loguru setup; stdlib `logging` removed |
| `services/reranker/reranker_app.py` | Minimal loguru setup; `RerankRequest` adds `instruction: str \| None = None` |
| `services/guard/guard_app.py` | Minimal loguru setup |

### Root
| File | Change |
|---|---|
| `docker-compose.yml` | Build context `./orchestrator` → `./api`; add `EMBEDDING_MODEL`, `RERANKER_MODEL`, `GUARD_MODEL` env vars |
| `.env.example` | Rewritten with nested-delimiter vars; stale flat vars removed |
| `pyproject.toml` | Add `loguru>=0.7.0` |
| `api/requirements.txt` | Add `loguru==0.7.3` |
| `tests/orchestrator/conftest.py` | `_root / "orchestrator"` → `_root / "api"` |
| `tests/orchestrator/test_config.py` | Update access patterns; add sub-group unit tests |

---

## 8. Dependency Addition

```toml
# pyproject.toml
"loguru>=0.7.0",
```

```txt
# api/requirements.txt  AND  each service requirements.txt
loguru==0.7.3
```

---

## 9. Testing Strategy

- **Config unit tests** (`test_config.py`): instantiate each sub-group directly with
  dict kwargs — no env setup needed. Assert defaults and nested access paths.
  ```python
  from orchestrator.config import RAGConfig
  cfg = RAGConfig()
  assert cfg.retrieval_top_k == 20
  assert cfg.context_limit == 5
  ```
- **Integration tests**: update all `s.retrieval_top_k` → `s.rag.retrieval_top_k`
  access patterns. Existing mock-based tests pass without other changes.
- **Log capture**: loguru writes to stderr; pytest's `capsys` captures it.
  Domain level tests verify `logger.log("RETRIEVAL", ...)` emits at correct level.

---

## 10. Rollout Order

1. `git mv orchestrator/ api/`; update `docker-compose.yml` build context
2. Fix `tests/orchestrator/conftest.py` sys.path (`"orchestrator"` → `"api"`)
3. Rewrite `api/orchestrator/config.py` with 5 nested groups + `SettingsConfigDict`
4. Rewrite `api/orchestrator/tracing.py` — `setup_logging` + `setup_langsmith` + domain levels
5. Update `api/orchestrator/main.py` — call both setup functions; update `LLMClient` construction
6. Update `api/orchestrator/pipeline/rag.py` — prompts + limits from settings; `user_template.format()`
7. Update all `api/orchestrator/services/*.py` — settings access + loguru domain levels
8. Update `api/orchestrator/routers/chat.py` — `logger.contextualize(request_id=...)`
9. Update `services/*/requirements.txt` — add `loguru==0.7.3`
10. Update `services/*_app.py` — minimal loguru setup; reranker adds `instruction` field
11. Update `.env.example`, `pyproject.toml`, `api/requirements.txt`
12. Update `tests/` — config access patterns
13. Run `uv run pytest tests/ --tb=short` — all tests must pass
14. `docker compose build && docker compose up -d` — verify all containers healthy
