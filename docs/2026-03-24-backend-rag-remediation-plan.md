# Backend RAG Remediation Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Sửa backend để có hybrid retrieval đúng nghĩa, hội thoại nhiều turn có context, citations chính xác kể cả khi dùng web search fallback, inline markdown link citations dạng `[title](url)` thay cho list cuối câu trả lời, streaming SSE giữ nguyên định dạng mà model sinh ra, non-streaming/streaming contract gần với OpenAI chuẩn hơn, LangSmith hiển thị output đúng, và VRAM ổn định hơn mà vẫn giữ low latency cho text chat.

**Architecture:** OpenWebUI chỉ gửi `request.messages` và hiển thị kết quả. Backend chịu trách nhiệm toàn bộ cho task classification, history summarization, follow-up query rewriting, sparse query encoding, provenance-aware citation generation, backend-owned citation placeholder rendering sang inline markdown links `[title](url)`, streaming transport giữ nguyên nguyên văn `answer`, OpenAI-compatible response shaping, LangSmith trace shaping, và GPU memory instrumentation. Không thêm logic sản phẩm ở frontend và không yêu cầu schema mới từ OpenWebUI cho phase remediation này; phần renderer/proxy chỉ được chỉnh sau khi backend đã chứng minh trả đúng nội dung.

**Tech Stack:** FastAPI, LangGraph, LangSmith, Qdrant, Tavily, SentenceTransformers, Transformers, PyTorch, underthesea, fastembed BM25, pytest.

---

## Locked Decisions

- OpenWebUI chỉ là interface; toàn bộ quyết định nghiệp vụ và citation logic phải nằm ở backend.
- History policy: `summary + recent`.
- Ngưỡng mặc định:
  - summarize khi có hơn `6` non-system messages,
  - giữ nguyên `3` non-system messages cuối trong prompt.
- summarize threshold ở trên chỉ là default; backend phải xét thêm token budget của prompt, không được cưỡng bức một extra summarization call trên mọi turn dài nếu prompt vẫn nằm trong ngân sách.
- `resolved_query` là input chuẩn cho input guard, embedding, sparse encoding, retrieval, web fallback, output guard.
- `raw_query` là input chuẩn cho UX và câu trả lời cuối cùng.
- VRAM policy: `low latency first`.
- `embedding`, `reranker`, `guard` vẫn resident trên GPU trong đợt này; tối ưu theo request và observability thay vì chuyển sang on-demand unload.
- Citation target:
  - backend phải render inline markdown links dạng `[title](url)` trực tiếp trong answer,
  - backend phải dùng backend-owned placeholders nội bộ rồi mới render ra link markdown cuối cùng,
  - citation output không dùng numeric footnotes `[1]`,
  - không append bullet list `Nguồn:` ở cuối như hiện tại.
- Nếu nguồn không có URL hợp lệ, fallback dạng plain text `title (source)` thay vì tạo markdown link hỏng.
- Renderer compatibility target:
  - chuẩn canonical là inline markdown links `[title](url)`,
  - không fallback về numeric footnotes,
  - nếu renderer có vấn đề với markdown links thì xử lý ở transport/proxy/render diagnosis chứ không quay lại numbered citations.
- Streaming transport target:
  - SSE chunking không được chuẩn hóa whitespace,
  - nội dung ghép lại từ toàn bộ `choices[0].delta.content` phải đúng bằng `answer` backend đã sinh ra,
  - không dùng `split()` theo whitespace hoặc bất kỳ chiến lược pseudo-streaming nào làm mất `\n`, dòng trống, indent, hay markdown spacing.
- API contract target:
  - non-streaming response dùng `message.content` thay vì chỉ `delta.content`,
  - streaming response tiếp tục dùng `delta.content`,
  - canonical backend contract phải được test rõ thay vì chỉ dựa vào tolerance của OpenWebUI.
- Context model target:
  - ngoài provenance tối thiểu, context items nên mang sẵn `collection_name`, `doc_type`, `chunk_type`, `modality`, `source_quality`, `image_path`, `structured_data` khi có thể để tránh một đợt refactor thứ hai ở phase multimodal/multi-collection.
- Reproducibility target:
  - local test entrypoint chuẩn là `uv run pytest`,
  - integration tests phải là opt-in chứ không fail trong môi trường local mặc định,
  - remediation implementation phải leave repo ở trạng thái full local unit suite reproducible.
- Compatibility validation policy:
  - chỉ nghi ngờ OpenWebUI sau khi đã chứng minh non-streaming và streaming từ backend cho cùng một `answer` là giống nhau,
  - kiểm chứng renderer trên các image version được pin cố định, không dựa vào image tag trôi nổi như `:main` hay `:latest`,
  - nếu còn mismatch sau khi backend đã preserve nguyên văn output, mới điều tra proxy buffering, compression, hoặc thiết lập chunk batching của OpenWebUI.

## Task 0: Baseline Evaluation, Test Entry Point, And Contract Guardrails

**Files:**

- Create: `tests/orchestrator/test_regression_matrix.py`
- Modify: `tests/services/conftest.py`
- Modify: `pyproject.toml`
- Modify: `Makefile`
- Modify: `README.md`

- [ ] **Step 1: Add a baseline regression matrix before refactoring behavior**

Add a focused regression matrix covering:

- single-turn factual text queries,
- follow-up queries that require prior-turn context,
- keyword-sensitive / entity-heavy queries where sparse retrieval should matter,
- no-context queries,
- web-fallback-needed queries,
- citation-sensitive answers where inline links must be preserved through streaming.

Goal:

- capture current behavior before remediation,
- create a stable acceptance harness for later tasks,
- avoid finishing "green" on unit tests while still regressing user-facing answer quality.

- [ ] **Step 2: Normalize the local test workflow**

Update repo-level workflow so that:

- canonical local command is `uv run pytest`,
- `integration` tests are explicitly marked/registered and skipped unless requested,
- service tests do not fail only because root dev environment is missing mockable libraries such as `sentence_transformers`.

- [ ] **Step 3: Add repo-facing documentation for the test contract**

Document in `README.md`:

- which command is the canonical local unit-test entrypoint,
- which tests require Docker/GPU/integration setup,
- what part of the remediation plan is measured by the regression matrix.

- [ ] **Step 4: Run focused verification**

Run:

```bash
uv run pytest tests/orchestrator/test_regression_matrix.py tests/services/test_embedding.py tests/services/test_asr.py tests/services/test_tts.py -q
```

Expected: PASS locally without requiring the full Docker stack.

## Task 1: Backend Conversation Context, Request Contract, And Task Routing

**Files:**

- Create: `api/orchestrator/services/conversation.py`
- Modify: `api/orchestrator/routers/chat.py`
- Modify: `api/orchestrator/pipeline/rag.py`
- Modify: `api/orchestrator/models/schemas.py`
- Test: `tests/orchestrator/test_chat_router.py`
- Test: `tests/orchestrator/test_rag_pipeline.py`
- Test: `tests/orchestrator/test_conversation.py`
- Test: `tests/orchestrator/test_schemas.py`

- [ ] **Step 1: Write failing tests for multi-turn backend context handling**

Add tests for:

- main chat no longer discards earlier turns,
- follow-up question is rewritten into a standalone `resolved_query`,
- summarization is skipped for short histories,
- summarization runs for histories above threshold,
- auxiliary tasks are classified in backend without relying on frontend behavior,
- non-streaming contract returns canonical `message.content`,
- streaming contract continues to use `delta.content`.

Run:

```bash
pytest tests/orchestrator/test_chat_router.py tests/orchestrator/test_rag_pipeline.py -q
```

Expected: FAIL because current backend only extracts the last user message.

- [ ] **Step 2: Add a conversation service owned by backend**

Implement backend helpers in `api/orchestrator/services/conversation.py`:

- `normalize_messages(messages) -> normalized conversation`
- `extract_latest_user_query(messages) -> raw_query`
- `should_summarize(messages) -> bool`
- `summarize_history(messages) -> summary text`
- `build_conversation_context(summary, recent_turns) -> prompt block`
- `rewrite_followup_query(raw_query, summary, recent_turns) -> resolved_query`
- `classify_task(messages, latest_user_message) -> task_type`
- `build_auxiliary_history(messages, latest_user_message) -> history string`

Rules:

- ignore `system` messages when counting turns for summary threshold,
- keep `system` messages available for future prompt composition but do not trust frontend to decide retrieval behavior,
- summarization gate should consider prompt token budget in addition to raw message count,
- use backend-generated `resolved_query` everywhere retrieval/guard logic needs a standalone question,
- preserve backward compatibility for current OpenWebUI task prompts by parsing `<chat_history>` only when `request.messages` does not contain enough prior turns.

Update `api/orchestrator/models/schemas.py` so that:

- non-streaming response shape is clearly distinct from streaming chunk shape,
- current text-only request path remains backward compatible,
- the schema keeps a clean extension point for future structured content parts without forcing a second total rewrite later.

- [ ] **Step 3: Rewire router and pipeline to use backend-managed conversation state**

Change `api/orchestrator/routers/chat.py`:

- stop calling `rag_graph.ainvoke(_build_initial_state(user_message))`,
- build a backend request payload that includes `raw_query`, `resolved_query`, `conversation_context`, and `task_type`,
- route `title`, `tags`, and `follow_ups` via backend task dispatch,
- normalize non-streaming output to canonical OpenAI-like `message.content` while preserving streaming `delta.content`.

Change `api/orchestrator/pipeline/rag.py`:

- extend state to include:
  - `raw_query`
  - `resolved_query`
  - `conversation_summary`
  - `conversation_context`
  - `task_type`
- use `resolved_query` in:
  - input guard,
  - embedding,
  - sparse encoding,
  - retrieval,
  - web fallback,
  - output guard.
- use `conversation_context + final_context + raw_query` when building the final generation prompt.

- [ ] **Step 4: Run tests and verify no regression on single-turn chat**

Run:

```bash
pytest tests/orchestrator/test_chat_router.py tests/orchestrator/test_rag_pipeline.py tests/orchestrator/test_schemas.py -q
```

Expected: PASS with both single-turn and multi-turn flows covered.

- [ ] **Step 5: Commit**

```bash
git add api/orchestrator/services/conversation.py api/orchestrator/routers/chat.py api/orchestrator/pipeline/rag.py api/orchestrator/models/schemas.py tests/orchestrator/test_chat_router.py tests/orchestrator/test_rag_pipeline.py tests/orchestrator/test_conversation.py
git commit -m "feat: move conversation logic fully into backend"
```

## Task 2: Real Hybrid Retrieval With Sparse Query Encoding

**Files:**

- Create: `api/orchestrator/services/sparse_encoder.py`
- Modify: `api/orchestrator/pipeline/rag.py`
- Modify: `api/orchestrator/main.py`
- Modify: `api/requirements.txt`
- Modify: `tests/orchestrator/test_retriever.py`
- Modify: `tests/orchestrator/test_rag_pipeline.py`
- Test: `tests/orchestrator/test_sparse_encoder.py`

- [ ] **Step 1: Write failing tests that prove sparse is not wired end-to-end**

Add tests for:

- sparse encoder returns `indices/values`,
- pipeline passes non-`None` `sparse_vector` into `RetrieverClient.hybrid_search()`,
- fallback to dense-only when sparse encoder errors.

Run:

```bash
pytest tests/orchestrator/test_retriever.py tests/orchestrator/test_rag_pipeline.py -q
```

Expected: FAIL because current pipeline never generates or passes a sparse query vector.

- [ ] **Step 2: Implement query-side sparse encoder in backend**

Add `api/orchestrator/services/sparse_encoder.py` with:

- `tokenize_vietnamese(text)` using `underthesea.word_tokenize(..., format="text")`
- a singleton/lazy `Bm25("Qdrant/bm25")`
- `encode_query(text) -> {"indices": [...], "values": [...]}`.

Constraints:

- must use the same tokenization and BM25 model family as `scripts/ingest.py`,
- must stay in backend; do not move sparse query creation into OpenWebUI,
- must be resilient to initialization error and allow dense-only fallback.

- [ ] **Step 3: Wire sparse encoding into the RAG path**

Change `api/orchestrator/main.py` and `api/orchestrator/pipeline/rag.py`:

- register the new sparse encoder service,
- encode sparse from `resolved_query`,
- pass both dense and sparse vectors to `hybrid_search()`,
- log whether retrieval executed in `hybrid` or `dense_only` mode,
- preserve current Qdrant vector names `dense` and `sparse`.

- [ ] **Step 4: Update dependencies and run focused tests**

Add backend dependencies needed for sparse encoding to `api/requirements.txt`.

Run:

```bash
pytest tests/orchestrator/test_sparse_encoder.py tests/orchestrator/test_retriever.py tests/orchestrator/test_rag_pipeline.py -q
```

Expected: PASS with explicit coverage for dense+sparse behavior.

- [ ] **Step 5: Commit**

```bash
git add api/orchestrator/services/sparse_encoder.py api/orchestrator/pipeline/rag.py api/orchestrator/main.py api/requirements.txt tests/orchestrator/test_sparse_encoder.py tests/orchestrator/test_retriever.py tests/orchestrator/test_rag_pipeline.py
git commit -m "feat: enable true hybrid retrieval with sparse query encoding"
```

## Task 3: Provenance-Aware Context Model For Citations

**Files:**

- Modify: `api/orchestrator/pipeline/rag.py`
- Modify: `api/orchestrator/services/web_search.py`
- Modify: `api/orchestrator/models/schemas.py`
- Test: `tests/orchestrator/test_rag_pipeline.py`
- Test: `tests/orchestrator/test_web_search.py`

- [ ] **Step 1: Write failing tests for web-search citation provenance**

Add tests for:

- when web fallback contributes results, the final citation candidates preserve `source_type=web`,
- citations are no longer derived by blindly slicing the first `citation_limit` items from `final_context`,
- web results remain traceable after combine/merge.

Run:

```bash
pytest tests/orchestrator/test_rag_pipeline.py tests/orchestrator/test_web_search.py -q
```

Expected: FAIL because current `final_context = reranked + web_results` and `citations_node` slices from the head of that list.

- [ ] **Step 2: Enrich context items with provenance metadata**

Extend every context item produced by retrieval/web fallback to carry:

- `context_id`
- `source_type` with values `hybrid` or `web`
- `retrieval_stage`
- `original_rank`
- `collection_name`
- `doc_type`
- `chunk_type`
- `modality`
- `source_quality`
- `title`
- `url`
- `source`
- `score`
- `image_path`
- `structured_data`

Rules:

- `context_id` must be stable for the lifetime of one request and be the only id used in generation/citation placeholders,
- `WebSearchClient.search()` must tag every result with `source_type="web"`,
- hybrid results must be tagged `source_type="hybrid"`.

- [ ] **Step 3: Build a citation pool separate from list order**

Refactor `api/orchestrator/pipeline/rag.py` so that:

- `final_context` is still the content pool for generation,
- `citation_pool` is built from stable `context_id`-keyed entries, not from positional slicing,
- `citations_node` no longer assumes the first `N` items are the correct citations,
- every citation returned downstream is resolved from `context_id`.

Do not rely on:

- list order,
- `citation_limit` applied before provenance is resolved,
- `reranked + web_results` ordering as a proxy for what the answer actually used.

- [ ] **Step 4: Run focused tests**

Run:

```bash
pytest tests/orchestrator/test_rag_pipeline.py tests/orchestrator/test_web_search.py -q
```

Expected: PASS with explicit assertions for `source_type` and provenance fields.

- [ ] **Step 5: Commit**

```bash
git add api/orchestrator/pipeline/rag.py api/orchestrator/services/web_search.py api/orchestrator/models/schemas.py tests/orchestrator/test_rag_pipeline.py tests/orchestrator/test_web_search.py
git commit -m "refactor: track citation provenance across hybrid and web context"
```

## Task 4: Inline Markdown Link Citations Instead Of End-Appended Source List

**Files:**

- Modify: `api/orchestrator/pipeline/rag.py`
- Modify: `api/orchestrator/routers/chat.py`
- Modify: `api/orchestrator/services/llm.py`
- Modify: `tests/orchestrator/test_chat_router.py`
- Modify: `tests/orchestrator/test_rag_pipeline.py`
- Test: `tests/orchestrator/test_inline_citations.py`

- [ ] **Step 1: Write failing tests for inline link citation rendering**

Add tests for:

- generation prompt contains stable `context_id` references and source metadata,
- answer text contains inline markdown links `[title](url)` rendered by backend,
- reconstructed streaming content matches the exact `answer` string, including `\n`, blank lines, markdown list markers, and emphasis markers,
- streaming response no longer appends a markdown bullet list under `**Nguồn:**`,
- web-search answers can reference web entries by rendered markdown links,
- invalid/missing citations trigger repair flow.

Run:

```bash
pytest tests/orchestrator/test_chat_router.py tests/orchestrator/test_rag_pipeline.py -q
```

Expected: FAIL because current router appends a citation list chunk at the end of the answer stream and currently collapses formatting by tokenizing with whitespace splitting.

- [ ] **Step 2: Introduce backend-owned citation placeholders in prompting**

Refactor context formatting in `api/orchestrator/pipeline/rag.py`:

- assign each context entry a stable `context_id`,
- inject both snippet text and source metadata into the prompt,
- instruct the LLM to cite factual claims inline using backend placeholders such as `[[cite:context_id]]`,
- keep placeholder generation backend-owned; frontend must not contribute citation ids or numbering.

Prompt rules:

- if the answer uses a source, place the placeholder immediately after the claim,
- if a claim is supported by multiple sources, allow multiple placeholders,
- do not invent ids not present in the prompt context.

- [ ] **Step 3: Add citation validation, repair, and markdown rendering in backend**

Implement a post-generation step:

- parse cited placeholders from the draft answer,
- validate every placeholder id against `citation_pool`,
- if the answer contains no valid placeholders, run one repair pass that adds placeholders to the existing answer without changing factual content,
- if repair still fails, fallback to deterministic top-support citations from the prompt context and log the fallback,
- render validated placeholders into final inline markdown links `[title](url)`,
- if a citation has no valid URL, fallback to `title (source)` instead of broken markdown.

This step must remain entirely in backend.

- [ ] **Step 4: Switch transport formatting from list-tail citations to link-aware answer content**

Change `api/orchestrator/routers/chat.py`:

- stop appending a separate `**Nguồn:**` chunk for streaming,
- replace whitespace-based pseudo-streaming with chunk slicing that preserves the original string exactly,
- stream the answer text exactly as produced after citation post-processing,
- for non-streaming, return answer content via canonical `message.content`.

Transport rules:

- chunk boundaries may split anywhere safe in the string, but must not rewrite content,
- reconstructed streamed content must equal the post-processed `answer` byte-for-byte at Python `str` level,
- if fine-grained chunking complicates correctness, prefer fewer larger chunks over lossy token splitting.

Rendering target:

- canonical format is inline markdown links `[title](url)`,
- do not fallback to numeric references or trailing source blocks.

- [ ] **Step 5: Verify renderer compatibility before locking final output format**

Add one validation task during implementation:

- compare non-streaming and streaming responses for the same multiline markdown answer and confirm they are identical after concatenating stream deltas,
- test the resulting answer in the running OpenWebUI renderer,
- if markdown links render correctly, keep canonical format as `[title](url)`,
- if not, diagnose renderer/proxy behavior without changing the canonical citation style.

Only after backend parity is confirmed:

- test whether OpenWebUI still alters rendering,
- if yes, inspect OpenWebUI settings for stream chunk batching and any reverse-proxy buffering/compression in front of it.

This validation changes only formatting strategy and compatibility diagnosis, not backend ownership.

- [ ] **Step 6: Run focused tests**

Run:

```bash
pytest tests/orchestrator/test_chat_router.py tests/orchestrator/test_rag_pipeline.py tests/orchestrator/test_inline_citations.py -q
```

Expected: PASS with end-of-answer citation list removed.

- [ ] **Step 7: Commit**

```bash
git add api/orchestrator/pipeline/rag.py api/orchestrator/routers/chat.py api/orchestrator/services/llm.py tests/orchestrator/test_chat_router.py tests/orchestrator/test_rag_pipeline.py tests/orchestrator/test_inline_citations.py
git commit -m "feat: replace trailing citation list with inline markdown link citations"
```

## Task 5: LangSmith Root Output Normalization

**Files:**

- Modify: `api/orchestrator/routers/chat.py`
- Modify: `api/orchestrator/pipeline/rag.py`
- Test: `tests/orchestrator/test_chat_router.py`
- Test: `tests/orchestrator/test_tracing.py`

- [ ] **Step 1: Write failing tests for root output shape**

Add tests proving that:

- root traced output should expose `answer` and resolved citations only,
- `generation_prompt` must not appear in root output,
- raw graph state should not be the display payload used by LangSmith list view.

Run:

```bash
pytest tests/orchestrator/test_chat_router.py tests/orchestrator/test_tracing.py -q
```

Expected: FAIL because current traced root output is effectively the raw graph state or raw router result.

- [ ] **Step 2: Add a trace wrapper dedicated to backend request execution**

Implement a root-level backend function such as `execute_chat_turn()`:

- mark it `@traceable`,
- return a compact output contract:
  - `answer`
  - `citations`
  - `task_type`
  - `resolved_query`
- keep `LangGraph` child runs available for deep debugging but not as the root display payload.

- [ ] **Step 3: Run focused tests**

Run:

```bash
pytest tests/orchestrator/test_chat_router.py tests/orchestrator/test_tracing.py -q
```

Expected: PASS with trace output contract stabilized.

- [ ] **Step 4: Commit**

```bash
git add api/orchestrator/routers/chat.py api/orchestrator/pipeline/rag.py tests/orchestrator/test_chat_router.py tests/orchestrator/test_tracing.py
git commit -m "fix: normalize backend trace output for LangSmith"
```

## Task 6: OpenWebUI And Infra Compatibility Hardening After Backend Stream Fix

**Files:**

- Modify: `docker-compose.yml`
- Modify: `docker-compose.dev.yaml`
- Modify: `docker-compose.vast.yaml`
- Modify: `docs/vast-ai-deployment.md`
- Test: `tests/orchestrator/test_chat_router.py`

- [ ] **Step 1: Add an explicit regression test for transport parity**

Add one focused chat-router test proving that for a multiline markdown answer:

- non-streaming returns the exact original string,
- streaming returns chunks whose concatenated `delta.content` is exactly the same string,
- no extra `Nguồn:` tail block is injected by transport.

Run:

```bash
pytest tests/orchestrator/test_chat_router.py -q
```

Expected: FAIL until the stream transport preserves exact formatting.

- [ ] **Step 2: Pin all moving infra images needed for reproducible diagnosis**

Change `docker-compose.yml`, `docker-compose.dev.yaml`, and `docker-compose.vast.yaml`:

- stop using `ghcr.io/open-webui/open-webui:main`,
- pin to a specific tested OpenWebUI release tag,
- pin other relevant moving images such as `qdrant` and `cloudflared` to known-good versions,
- keep the tag centralized via env var or a clearly documented literal so future upgrades are intentional.

Rule:

- do not diagnose renderer behavior against a moving target image.

- [ ] **Step 3: Validate running OpenWebUI against the fixed backend**

Manual verification checklist:

- open the running OpenWebUI instance wired to this backend,
- send a prompt that produces paragraphs, bullet lists, and bold markdown,
- compare the rendered result for `stream=false` versus `stream=true`,
- record whether renderer output is now correct once backend transport parity is fixed.

If mismatch still exists:

- inspect OpenWebUI stream batching settings,
- inspect reverse-proxy buffering/compression in front of OpenWebUI,
- capture the raw SSE payload to prove whether corruption happens before or after UI ingestion.

- [ ] **Step 4: Document the compatibility outcome**

Update `docs/vast-ai-deployment.md` with:

- the pinned OpenWebUI version,
- any required proxy settings for SSE,
- the known-good verification procedure for multiline markdown streaming.

- [ ] **Step 5: Run focused verification**

Run:

```bash
pytest tests/orchestrator/test_chat_router.py -q
```

Expected: PASS, plus manual confirmation that OpenWebUI renders the preserved streamed markdown correctly or an isolated note that the remaining issue is outside backend transport.

- [ ] **Step 6: Commit**

```bash
git add docker-compose.yml docker-compose.dev.yaml docker-compose.vast.yaml docs/vast-ai-deployment.md tests/orchestrator/test_chat_router.py
git commit -m "chore: pin openwebui and document stream compatibility checks"
```

## Task 7: Low-Latency VRAM Optimization And GPU Observability

**Files:**

- Modify: `services/embedding/embedding_app.py`
- Modify: `services/reranker/reranker_app.py`
- Modify: `services/guard/guard_app.py`
- Modify: `docker-compose.yml`
- Modify: `docker-compose.dev.yaml`
- Modify: `tests/services/test_reranker.py`
- Modify: `tests/services/test_guard.py`
- Modify: `tests/services/test_embedding.py`

- [ ] **Step 1: Write failing tests for memory-management contracts**

Add tests for:

- `torch.inference_mode()` usage in `guard` and `reranker`,
- cleanup hooks for intermediate tensors,
- `use_cache=False` on guard generation,
- memory metrics logging hooks being called.

Run:

```bash
pytest tests/services/test_guard.py tests/services/test_reranker.py tests/services/test_embedding.py -q
```

Expected: FAIL because current services do not expose these contracts.

- [ ] **Step 2: Add request-level GPU observability**

Implement lightweight helpers to log on each request:

- `memory_allocated`
- `memory_reserved`
- `max_memory_allocated`
- latency

Services to update:

- `embedding`
- `reranker`
- `guard`

Goal:

- distinguish real leaks from allocator high-water mark behavior,
- create evidence for plateau after warmup,
- keep instrumentation inside backend services only.

- [ ] **Step 3: Apply low-latency memory optimizations**

Change `services/guard/guard_app.py`:

- use `torch.inference_mode()`,
- set `use_cache=False`,
- reduce default classification `max_new_tokens` to `64`,
- delete intermediate tensors after each request.

Change `services/reranker/reranker_app.py`:

- use `torch.inference_mode()`,
- delete `inputs` and `logits` after each passage,
- keep sequential scoring to avoid peak VRAM spikes from large batches.

Change `services/embedding/embedding_app.py`:

- keep resident model,
- preserve OOM retry path,
- log memory before and after encode,
- keep cache cleanup only where it helps fragmentation instead of treating it as proof of leak fix.

Change compose:

- add `PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True` for `guard` and `reranker`,
- expose `GUARD_MAX_NEW_TOKENS`,
- tune dev defaults conservatively if profiling shows spikes.

- [ ] **Step 4: Run focused tests**

Run:

```bash
pytest tests/services/test_guard.py tests/services/test_reranker.py tests/services/test_embedding.py tests/services/test_compose_dev.py tests/services/test_dockerfiles.py -q
```

Expected: PASS with new env and memory behaviors covered.

- [ ] **Step 5: Commit**

```bash
git add services/embedding/embedding_app.py services/reranker/reranker_app.py services/guard/guard_app.py docker-compose.yml docker-compose.dev.yaml tests/services/test_guard.py tests/services/test_reranker.py tests/services/test_embedding.py
git commit -m "perf: stabilize gpu memory usage for text backend services"
```

## Task 8: Final Verification And Documentation Sync

**Files:**

- Modify: `docs/diagrams/architecture_sequence_diagram.mmd`
- Modify: `docs/diagrams/05_hybrid_retrieval_flow.mmd`
- Modify: `docs/diagrams/02_rag_pipeline_state_machine.mmd`
- Modify: `docs/diagrams/10_multimodal_rag_pipeline_state_machine.mmd`

- [ ] **Step 1: Update diagrams so docs match actual backend design**

Reflect the new behavior:

- full backend conversation logic,
- true sparse query path,
- provenance-aware citations,
- inline markdown link citations instead of trailing source list,
- LangSmith output normalization,
- VRAM instrumentation notes.

- [ ] **Step 2: Run the full targeted test matrix**

Run:

```bash
pytest tests/orchestrator/test_chat_router.py tests/orchestrator/test_conversation.py tests/orchestrator/test_sparse_encoder.py tests/orchestrator/test_retriever.py tests/orchestrator/test_rag_pipeline.py tests/orchestrator/test_web_search.py tests/orchestrator/test_inline_citations.py tests/orchestrator/test_tracing.py tests/services/test_guard.py tests/services/test_reranker.py tests/services/test_embedding.py -q
```

Expected: PASS.

- [ ] **Step 3: Do one manual validation pass in the running stack**

Validate:

- follow-up question uses prior context correctly,
- hybrid retrieval logs `hybrid` mode and produces sparse hits,
- web fallback answer cites web sources when relevant,
- answer renders inline markdown links in OpenWebUI,
- LangSmith output column shows final answer, not raw query or generation prompt,
- repeated text queries show plateauing VRAM metrics after warmup.

- [ ] **Step 4: Commit**

```bash
git add docs/diagrams/architecture_sequence_diagram.mmd docs/diagrams/05_hybrid_retrieval_flow.mmd docs/diagrams/02_rag_pipeline_state_machine.mmd docs/diagrams/10_multimodal_rag_pipeline_state_machine.mmd
git commit -m "docs: align architecture diagrams with backend remediation plan"
```

## Acceptance Criteria

- Multi-turn chat uses backend-managed context and no longer discards prior turns.
- Auxiliary tasks still work, but the logic lives in backend and does not depend on frontend to decide behavior.
- Hybrid retrieval actually sends both dense and sparse query vectors.
- Web fallback citations preserve `source_type=web` and are no longer replaced by head-of-list hybrid citations.
- Answer output uses inline markdown links `[title](url)` and no longer appends a trailing bullet list of citations.
- Non-streaming contract uses canonical `message.content` while streaming continues to use `delta.content`.
- LangSmith list view shows final answer as output.
- Text services remain low-latency while GPU memory growth becomes measurable, explainable, and bounded after warmup.

## Explicit Out-Of-Scope For This Plan

- Adding a persistent conversation store outside `request.messages`.
- Moving any product logic into OpenWebUI.
- Reworking ASR/TTS orchestration in the main backend request path beyond memory-readiness and future compatibility.
- Implementing full multimodal image/audio request handling in this remediation plan; this plan only prepares the backend contract so later multimodal phases do not require another full response/citation refactor.
