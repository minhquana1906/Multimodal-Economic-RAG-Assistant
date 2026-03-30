# RAG Web Fallback And Query Rewrite Design

**Date:** 2026-03-30

## Goal

Refine the orchestrator answer pipeline so that:

- every user question is rewritten into a complete, well-formed search/query form before retrieval and web search;
- internal documents remain the primary source of truth;
- web search is added automatically when internal evidence is weak, using a two-threshold policy;
- answers do not fall back to default refusal or apology text when internal documents are missing;
- web-derived citations are returned as source URLs as the current citation format.

## Current Problems

1. The current `resolved_query` rewrite is mostly string concatenation with conversation history, which makes follow-up and web-search queries noisy and often low quality.
2. The current web fallback only uses a single heuristic based on minimum chunk count and a single score threshold.
3. When no final context exists, the pipeline returns a default apology-style `no_context_message` instead of attempting a more useful neutral answer path.
4. Prompting does not clearly instruct the model to combine internal and web sources while preserving source attribution by URL for web evidence.

## Proposed Architecture

### 1. Unified Query Normalization For Every Request

Introduce a unified query normalization step that runs for every chat request before retrieval:

- Input: `raw_query`, conversation summary, recent turns.
- Output: a single `resolved_query` that is complete, grammatically correct, explicit, and suitable for retrieval, reranking, and web search.

Behavior:

- Always run, not only for follow-up turns.
- Prefer an LLM-backed rewrite prompt that preserves user intent, fixes spelling, resolves references, and compresses relevant context into one standalone query.
- If rewrite fails, returns empty text, or errors, fall back to deterministic rule-based rewriting using the existing conversation formatting utilities.

This normalized query becomes the canonical query for:

- input guard;
- embedding generation;
- retrieval;
- reranking;
- web search;
- output guard prompt metadata.

The original `raw_query` is still preserved for final answering so the assistant responds to the user’s original wording.

### 2. Two-Threshold Web Fallback Policy

Replace the single-threshold fallback decision with a two-threshold policy:

- Hard threshold: `< 0.70` means always add web search.
- Soft threshold: `0.70 <= score < 0.85` means conditionally add web search.
- Strong internal evidence: `>= 0.85` means skip web search by default.

Soft-threshold fallback triggers when any of the following are true:

- reranked internal evidence is too shallow, such as only one strong chunk;
- score drops sharply after the first result, suggesting weak supporting evidence;
- the query appears time-sensitive or update-seeking;
- the normalized query was expanded materially from prior context, indicating ambiguity risk in internal retrieval.

This design avoids over-triggering web search at a flat `0.85` while still covering weak or ambiguous internal retrieval better than a flat `0.70`.

### 3. Combined Internal + Web Evidence

The final context assembly continues to support mixed sources:

- internal reranked chunks remain first in priority order;
- web results are appended when fallback triggers;
- citations are built from the combined provenance pool.

Answering behavior:

- if internal evidence is sufficient, answer mainly from internal documents;
- if internal evidence is partial, supplement with web evidence;
- if internal evidence is absent but web evidence exists, answer from web evidence;
- if both are absent, return a neutral “no data found” style response without apology wording.

### 4. Prompt And Response Rules

Update the generation prompt and prompt defaults so the model is instructed to:

- answer in Vietnamese;
- prioritize internal documents when they are sufficient;
- supplement with web evidence when internal coverage is incomplete;
- avoid default refusals or apology phrases when the issue is only missing data;
- cite web sources via their URLs in the returned citations/footer;
- avoid claiming unsupported facts.

Guard-related denials remain separate from missing-data behavior. Safety policy denials can still be explicit, but missing-context responses should be neutral and direct.

## Component Changes

### `api/orchestrator/services/conversation.py`

- Replace the current follow-up-only rewrite behavior with a general-purpose query normalization function for all requests.
- Keep deterministic helpers for formatting history and fallback rewriting.
- Add a rewrite prompt builder that converts summary plus recent context into one standalone query.

### `api/orchestrator/routers/chat.py`

- Update conversation preparation to call the new normalization function for every request.
- Preserve `raw_query`, `conversation_summary`, and `conversation_context` separately.

### `api/orchestrator/pipeline/rag.py`

- Replace current web fallback trigger logic with the two-threshold decision function.
- Use the normalized query for search-related operations.
- Adjust generation behavior so empty internal context no longer immediately returns an apology-style `no_context_message` if web evidence exists.
- Update generation prompt text to reflect combined-source behavior and neutral no-data handling.

### `api/orchestrator/config.py`

- Add separate hard and soft fallback thresholds or equivalent config names.
- Update prompt defaults to remove apology-style missing-data messages.
- Add rewrite prompt configuration if needed for maintainability.

### `api/orchestrator/services/llm.py`

- Reuse existing completion capabilities for query rewriting rather than introducing a new model client abstraction.

## Data Flow

1. User sends message.
2. Conversation layer extracts `raw_query`, summary, and recent turns.
3. Query normalization produces `resolved_query`.
4. Guard, embeddings, retrieval, and reranking use `resolved_query`.
5. Fallback policy evaluates rerank quality using the two-threshold logic.
6. If triggered, web search uses `resolved_query`.
7. Final context combines internal and web evidence.
8. Generation prompt answers the original user question using combined evidence.
9. Citations footer returns source links, with web citations pointing to source URLs.

## Error Handling

- Query rewrite failure:
    - log the failure;
    - fall back to deterministic rewrite;
    - if that also fails meaningfully, use `raw_query`.
- Web search failure:
    - do not fail the request;
    - continue with internal evidence only.
- No evidence anywhere:
    - return a neutral no-data message;
    - return empty citations.
- Guard rejection:
    - keep current safety path separate from missing-data behavior.

## Testing Strategy

Add or update tests for:

- normalization applied to first-turn questions, not only follow-ups;
- normalized query passed into guard, embedder, reranker, and web search;
- two-threshold fallback behavior:
    - below `0.70` always triggers;
    - between `0.70` and `0.85` triggers only under qualifying conditions;
    - at or above `0.85` does not trigger by default;
- mixed internal + web context produces combined citations;
- web-only answers include web URL citations;
- missing-data response is neutral and contains no apology/refusal wording;
- rewrite fallback path works when LLM rewrite fails.

## Open Implementation Choices

The main implementation choice is whether soft-threshold signals should be represented as:

- a compact boolean helper with explicit heuristics; or
- a weighted “confidence gap” score derived from top-k rerank statistics.

Recommendation:

- start with explicit heuristics because they are easier to reason about, test, and tune;
- only move to a weighted score if production traces show the heuristic policy is too coarse.

## Recommendation

Implement the change in one pass with tests first:

- add query normalization tests;
- add fallback decision tests;
- update prompt/config defaults;
- then wire the pipeline changes.

This keeps the retrieval semantics measurable while minimizing regression risk in the chat API surface.
