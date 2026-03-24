# Project Assessment And Roadmap Review

**Date:** 2026-03-24
**Scope reviewed:** current repository state, `docs/2026-03-24-backend-rag-remediation-plan.md`, existing phase/spec docs, compose/deployment files, core backend/services/tests.

## 1. Executive Summary

The project already has a solid skeleton:

- text RAG path exists end-to-end,
- dense+sparse ingestion exists,
- ASR and TTS services already exist as real code,
- dev/vast deployment paths exist,
- unit-test coverage is much better than a typical prototype.

But the current backend still has several contract-level issues that directly affect product correctness:

- conversation state is effectively single-turn,
- sparse retrieval exists at ingest time but is not wired at query time,
- citation generation is positional rather than provenance-aware,
- SSE streaming currently rewrites the answer text,
- the OpenAI response contract is only partially compatible,
- future multimodal/data-pipeline plans are partially stale relative to the codebase.

The remediation plan is directionally correct and should be kept, but it should be tightened in three ways:

1. add a baseline evaluation + acceptance layer before refactoring,
2. normalize the backend response contract more aggressively,
3. align later phases with the current repo reality instead of the older "build from scratch" assumptions.

## 2. Evidence Snapshot

Key findings observed from the current code:

- Router only extracts the latest `user` message and discards prior turns: `api/orchestrator/routers/chat.py:77-86`.
- Streaming uses `answer.split()` and appends a separate `**Nguồn:**` tail block: `api/orchestrator/routers/chat.py:99-150`.
- Query-side sparse encoding is not called from the pipeline even though retrieval supports it: `api/orchestrator/pipeline/rag.py:120-126`, `api/orchestrator/services/retriever.py:18-43`.
- Citations are sliced from `final_context[:citation_limit]` rather than resolved from actual answer usage: `api/orchestrator/pipeline/rag.py:215-221`.
- Web results are normalized but not tagged with provenance fields like `source_type` or `context_id`: `api/orchestrator/services/web_search.py:39-47`.
- Request/response schemas are still text-only and not fully OpenAI-compatible:
  - `Message.content: str`: `api/orchestrator/models/schemas.py:10-20`
  - non-streaming responses also use `delta` instead of `message`: `api/orchestrator/models/schemas.py:23-47`, `api/orchestrator/routers/chat.py:91-97`
- The repo already has ASR/TTS service code:
  - `services/asr/asr_app.py`
  - `services/tts/tts_app.py`
  but later phase plans still describe creating these services from scratch.
- Vast compose already provisions a `vlm` runtime, but the backend has no `api/orchestrator/services/vlm.py` and no `services/vlm/` implementation.
- `README.md` is empty and root packaging metadata is still placeholder-level:
  - `README.md` has `0` lines
  - `pyproject.toml:4-5`
  - `main.py:1-6`

## 3. What Is Already Good

### 3.1 Repo shape is coherent

The project now has recognizable boundaries:

- `services/*` for GPU microservices,
- `api/orchestrator/*` for backend orchestration,
- `scripts/*` for ingestion,
- `docs/*` for specs/plans/deployment.

That makes the next round of refactoring realistic instead of chaotic.

### 3.2 The ingestion side is stronger than the serving side

`scripts/ingest.py` already tokenizes Vietnamese, builds sparse vectors, and writes dense+sparse vectors together into Qdrant. In other words, the data layer is ahead of the runtime query path.

This is good news because true hybrid retrieval can be unlocked without redesigning storage.

### 3.3 Deployment thinking is more mature than MVP prototypes usually are

The project already has:

- local dev compose with bind mounts,
- remote image-based Vast.ai compose,
- GPU placement strategy,
- Docker Hub push/pull flow,
- dedicated tests for compose and Dockerfile invariants.

That is a strong base for operational hardening.

## 4. Current Gaps And Improvement Opportunities

### 4.1 P0: Backend answer fidelity is not reliable enough yet

This is the most important issue because it affects the core product promise.

#### A. Conversation ownership is still in the wrong place

The backend still behaves as single-turn because the router extracts only the final user message and builds state from that string alone.

Impact:

- follow-up questions lose context,
- retrieval quality degrades on real chat sessions,
- any future citation correctness work is weakened because the query itself is under-specified.

#### B. "Hybrid retrieval" is incomplete at runtime

Dense+sparse indexing exists, and `RetrieverClient.hybrid_search()` supports a sparse vector, but the pipeline never produces one.

Impact:

- the system is not getting the lexical retrieval gains that the storage layer was built for,
- short, entity-heavy, ticker-heavy, and keyword-sensitive economic queries are likely underperforming.

#### C. Citation logic is structurally brittle

Today, citations are effectively "top N contexts after merge", not "sources actually grounded by the final answer".

Impact:

- citations can be wrong even when retrieval is right,
- web fallback provenance can be lost,
- future multimodal chunks will make this much worse because visual/table/chart chunks need richer attribution than title/url/score.

#### D. Streaming transport mutates content

Whitespace-splitting means the stream is not a faithful transport of the generated answer.

Impact:

- markdown layout can be corrupted,
- non-streaming vs streaming mismatch will be hard to reason about,
- OpenWebUI debugging becomes ambiguous because the backend itself is already lossy.

### 4.2 P0: The response contract should be normalized before more features are added

The backend is labeled OpenAI-compatible, but non-streaming responses still use `delta` instead of `message`.

Impact:

- compatibility depends on UI tolerance rather than API correctness,
- future multimodal/audio/image outputs become harder to integrate cleanly,
- debugging client behavior becomes harder because the contract is already custom.

Recommendation:

- keep streaming chunk schema OpenAI-style,
- switch non-streaming responses to true `message`,
- add contract tests that compare the backend payload shape against the OpenAI chat-completions contract used by OpenWebUI.

### 4.3 P1: The multimodal foundation is not ready at the schema layer

Current `Message.content` is a plain string, but the later plans assume OpenAI-style content parts for images and richer multimodal requests.

Impact:

- Phase 3A will require a schema break anyway,
- image/audio support cannot be bolted on cleanly without touching the request model,
- delaying this change increases refactor cost.

Recommendation:

- introduce content-part schemas early,
- support both legacy string content and structured content arrays during a transition period.

### 4.4 P1: Later phase plans are partially stale

Several plans still assume service creation work that is already done:

- Phase 2A still describes creating `services/asr/asr_app.py`,
- Phase 2B still describes creating `services/tts/tts_app.py`,
- current effort should move toward integration, API contract, and E2E verification instead.

There is also a VLM contract mismatch:

- older plans assume a custom `services/vlm/vlm_app.py` with task-specific endpoints,
- current Vast deployment provisions a generic VLM runtime via `vllm/vllm-openai`,
- the backend does not yet define which of those contracts it will actually consume.

Recommendation:

- choose one VLM contract now:
  - Option A: generic OpenAI-compatible multimodal chat endpoint through vLLM
  - Option B: custom backend-owned VLM microservice with `/analyze`, `/extract-table`, `/describe-chart`
- then rewrite Phase 3B and 3A around that decision.

### 4.5 P1: Reproducibility and DX still need tightening

Observed test behavior:

- `pytest -q` from the current shell failed immediately because local dependencies were not prepared for the bare interpreter,
- `uv run pytest -q` worked much better,
- full `uv run pytest -q` still showed two hygiene issues:
  - `tests/services/test_embedding.py` requires `sentence_transformers` in the dev environment,
  - `tests/services/test_integration.py` runs as a normal test and fails when Docker services are not up.

Recommendations:

- define one canonical test entrypoint, for example `make test` -> `uv run pytest ...`,
- either add `sentence-transformers` to the root dev extras or stub it in `tests/services/conftest.py`,
- register the `integration` marker and skip those tests unless explicitly requested,
- document the expected local workflow in `README.md`.

### 4.6 P1: Ops pinning should be broader than only OpenWebUI

The remediation plan correctly wants to pin OpenWebUI, but the stack still uses moving tags in other places such as:

- `qdrant/qdrant:latest`
- `cloudflare/cloudflared:latest`

Recommendation:

- pin all infrastructure images that affect runtime behavior,
- centralize image versions in env vars or a version matrix doc.

### 4.7 P2: Documentation and metadata lag behind implementation

The project is now beyond prototype stage, but repo metadata still looks prototype-grade:

- empty README,
- placeholder project description,
- placeholder root `main.py`.

This does not break runtime, but it slows onboarding and increases the chance of incorrect operator behavior.

## 5. Review Of The Remediation Plan

## 5.1 What the plan gets right

The plan is strong on the most important product decisions:

- backend owns business logic,
- `resolved_query` vs `raw_query` separation,
- provenance-aware citations,
- exact streaming parity,
- pinning the UI before diagnosing renderer issues,
- keeping latency-first VRAM policy.

Those are the right principles.

## 5.2 Changes I recommend before execution

### Add Task 0: Baseline evaluation and regression harness

The current remediation plan is contract-focused, but it is still missing a quality baseline.

Add a new Task 0 before Task 1:

- create a small gold set of Vietnamese economic questions:
  - single-turn,
  - follow-up/multi-turn,
  - entity/ticker-heavy,
  - ambiguous/no-context,
  - web-fallback-needed,
  - citation-sensitive prompts.
- log baseline for:
  - answer correctness,
  - retrieval mode (`dense_only` vs `hybrid`),
  - citation correctness,
  - latency per stage,
  - stream/non-stream answer parity.

Without this, the plan can finish "green" on unit tests but still not improve real answer quality enough.

### Split Task 1 into two sub-problems

Current Task 1 groups together:

- conversation normalization,
- summarization,
- follow-up rewriting,
- auxiliary task classification.

That is too much behavior change in one step.

Recommended split:

1. backend conversation extraction + task routing,
2. follow-up rewrite,
3. summarization policy.

This reduces blast radius and makes regressions easier to isolate.

### Make summarization token-budget aware, not only count-based

The fixed policy `>6 non-system messages, keep 3 recent` is a useful default, but it is not sufficient by itself.

Why:

- a long 4-message exchange may already exceed a useful token budget,
- a short 8-message exchange may not need summarization,
- if summary is recomputed from full history every turn, latency will grow.

Recommended improvement:

- keep the current threshold as a simple default,
- but gate summarization by estimated prompt budget too,
- cache summary artifacts in-memory or in Redis keyed by conversation hash if possible.

### Prefer structured citation placeholders over raw `[1]` generation

The current plan asks the LLM to directly emit inline human-readable footnotes and then repair them if invalid.

This works, but it is more fragile than necessary.

Recommended alternative:

- prompt the model to emit backend-owned placeholders such as `[[cite:ctx_3]]`,
- validate placeholders deterministically,
- only after validation render them to final `[1]` or `[^1]` format.

Benefits:

- simpler parsing,
- easier repair,
- less risk of false positives from numbers already present in the answer,
- cleaner support for future visual/table citations.

### Expand the context model now for later phases

Task 3 currently adds provenance fields, which is good, but future phases will need more than `source_type`.

Add fields now if possible:

- `collection_name`
- `doc_type`
- `chunk_type`
- `modality`
- `source_quality`
- `image_path`
- `structured_data`

This prevents a second citation/context refactor when multi-collection and VLM chunks arrive.

### Normalize OpenAI response compatibility explicitly

The plan should include an explicit contract-normalization step:

- non-streaming uses `message`,
- streaming uses `delta`,
- tests assert exact shape.

This belongs near Task 4/Task 6, but I would call it out explicitly rather than leaving it implicit.

### Move image pinning beyond OpenWebUI

Task 6 currently focuses on pinning OpenWebUI. Expand it to include all moving infrastructure images that matter for reproducible debugging.

## 6. Review Of Later Phases

## 6.1 The order should be updated to reflect current reality

A better near-term sequence now is:

1. Finish backend text-RAG remediation.
2. Normalize API/schema contracts for multimodal inputs and OpenAI compatibility.
3. Align the VLM service contract and implement the backend client.
4. Only then continue with multi-collection and crawl/PDF ingestion work.
5. Do full E2E last.

Reason:

- the text backend is still not stable enough to be the foundation for multimodal and multi-collection features,
- later phases currently assume stronger backend guarantees than the project actually has today.

## 6.2 ASR/TTS phases should be reframed as integration phases

ASR and TTS already exist as services. The remaining work is mostly:

- API contract integration into the chat endpoint,
- response shaping,
- orchestration behavior,
- E2E verification under real compose profiles.

So the future plan should stop talking about "create ASR/TTS service core" and instead talk about:

- "integrate existing ASR/TTS services into the chat contract",
- "add multimodal request/response models",
- "verify cold-start, unload, and audio UX behavior end-to-end".

## 6.3 VLM needs a single, stable product contract first

This is the biggest later-phase ambiguity.

Before touching image-query features, decide:

- whether the VLM is an OpenAI-compatible multimodal LLM endpoint,
- or a task-specific backend service with structured endpoints.

My recommendation:

- if the goal is runtime image understanding only, prefer the OpenAI-compatible vLLM path,
- if the goal is heavy structured extraction for offline ingestion, a backend-owned wrapper may still be useful,
- if both are needed, define them as two separate contracts on purpose rather than letting them drift into each other.

## 6.4 Data-pipeline plans are promising but optimistic

The crawl/PDF/VLM pipeline direction is good, but the current plans likely underestimate the operational work.

What is missing or under-specified:

- source legal/robots policies,
- parser drift and site-specific maintenance,
- dedup/versioning strategy across re-crawls,
- manifesting of raw/parsed/enriched/indexed artifacts,
- retry queues and partial-failure recovery,
- re-embedding/re-index versioning when chunking logic changes.

Recommendation:

- add a manifest-first design:
  - raw manifest,
  - parsed manifest,
  - chunk manifest,
  - index manifest,
- make every stage resumable by manifest status instead of only by ad-hoc file existence.

## 6.5 Phase 5 E2E is currently underspecified for the real integration cost

The current E2E phase assumes:

- stable multimodal request schema,
- stable VLM contract,
- multi-collection retrieval,
- output citation model for visual chunks,
- deployment/runtime observability.

These are not all ready yet.

Recommendation:

- split Phase 5 into:
  - 5A: contract and deployment verification
  - 5B: full modality E2E scenarios

## 7. Recommended Next Actions

### Immediate

- Keep the current remediation plan as the active path.
- Add a new Task 0 for baseline evaluation + regression harness.
- Add explicit response-contract normalization to the plan.
- Add richer provenance fields to the future citation/context model now.

### Near-term

- Rewrite stale phase docs for ASR/TTS from "build services" to "integrate services".
- Decide and document one VLM runtime contract.
- Introduce multimodal-compatible request schemas before Phase 3A work begins.

### Repo hygiene

- Write a real README with:
  - architecture,
  - local dev workflow,
  - test commands,
  - compose profiles,
  - deployment variants.
- Replace placeholder package metadata.
- Add canonical `make test` / `make test-unit` / `make test-integration`.

## 8. Verification Notes

Commands run during this review:

```bash
pytest -q
uv run pytest -q tests/orchestrator/test_chat_router.py tests/orchestrator/test_rag_pipeline.py tests/orchestrator/test_retriever.py tests/orchestrator/test_web_search.py tests/orchestrator/test_tracing.py tests/orchestrator/test_schemas.py tests/scripts/test_chunker.py tests/scripts/test_ingest.py tests/services/test_compose_dev.py tests/services/test_compose_vast.py tests/services/test_dockerfiles.py
uv run pytest -q tests/services/test_asr.py tests/services/test_tts.py
uv run pytest -q
```

Observed results:

- bare `pytest -q` failed immediately because the current shell environment was not the repo's prepared Python environment,
- focused `uv run pytest ...` subsets passed,
- full `uv run pytest -q` reached the end but still showed repo-level test-hygiene issues:
  - `tests/services/test_embedding.py` requires `sentence_transformers` in the root dev environment,
  - `tests/services/test_integration.py` is not isolated from non-Docker local runs.

That means the codebase is in a decent state, but the developer workflow and test contract still need one cleanup pass.
