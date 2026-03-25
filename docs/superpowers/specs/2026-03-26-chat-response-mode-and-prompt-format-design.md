# Chat Response Mode And Prompt Format Design

## Problem

Current answer formatting mixes concerns between the model output and backend citation rendering:

- The citation footer uses a different heading style from the rest of the answer and currently renders as bold text instead of a real section header.
- The main RAG text flow does not enforce a stable response structure, so answers can become long, inconsistent, or drift away from the user's question.
- Audio-origin questions currently share the same response style as text chat, which makes answers too long and too written for spoken playback.
- The chat request schema does not expose a clear marker that tells the backend whether the caller expects text-first formatting or voice-first formatting.

## Design

### Principle

The backend should explicitly control answer shape. The model still chooses content, but the backend defines:

- which response profile applies,
- which prompt instructions are used,
- when citation sections are appended, and
- which formatting contract the UI can rely on.

### A. Add An Explicit Top-Level Response Marker

Add a top-level `response_mode` field to the chat request contract:

- Allowed values: `"text"` and `"audio"`
- Default: `"text"`
- Scope: request-level only, not per-message metadata

Why this shape:

- It is explicit and easy to validate.
- It keeps backward compatibility for existing callers.
- It avoids scanning chat history or guessing from transport details.
- It gives the RAG pipeline a stable switch for prompt selection and final formatting.

### B. Text Mode Uses A Structured RAG Answer Contract

For `response_mode="text"`, the generation prompt should instruct the model to produce concise, structured markdown:

- Use only level-3 headers (`###`) for answer sections
- Separate sections with `----`
- Prefer short bullet points
- Lead with the direct answer
- Stay focused on the user's question
- Avoid long narrative paragraphs and avoid redundant conclusion sections

Recommended shape:

```md
### Trả lời ngắn gọn
- ...
- ...

----

### Phân tích chính
- ...
- ...

----

### Nguồn trích dẫn
- [Tiêu đề](url) - **source (score)**
```

The backend remains responsible for the citation footer so the final `Nguồn trích dẫn` section is deterministic even if the model varies slightly in the body.

### C. Citation Footer Uses The Same Header Level

When citations exist in text mode, the backend should append them as:

```md
----

### Nguồn trích dẫn
- [Tiêu đề](url) - **source (score)**
```

This replaces the current bold label footer and aligns citations with the rest of the answer structure.

The footer should continue to be generated from the backend citation pool rather than trusting the model to list sources correctly.

### D. Audio Mode Uses A Short Spoken-Style Answer Profile

For `response_mode="audio"`, the generation prompt should switch to a voice-first style:

- Return a single short paragraph
- Use direct spoken Vietnamese
- Answer the main question immediately
- Keep the answer to roughly 1-3 short sentences
- Avoid markdown headers, bullets, and decorative formatting
- Avoid invitation phrases, filler intros, and long caveats

The answer should sound natural when sent through TTS and should not force the listener through a long written-style explanation.

### E. Audio Mode Keeps Citations In Payload But Not In Spoken Text

For `response_mode="audio"`:

- Keep `citations` in the API payload for UI or debugging use
- Do not append the citation footer to `answer`

This prevents TTS from reading URLs, source names, and scores while preserving source metadata for the client.

### F. Scope Boundaries

This design changes answer shaping inside the chat/RAG path only.

Out of scope:

- Changing `/v1/audio/speech` or `/v1/audio/transcriptions` transport contracts
- Adding inline spoken citation synthesis
- Changing auxiliary non-RAG task prompts unless they later opt into `response_mode`
- Reworking streaming chunk transport beyond preserving the final answer text exactly

## Data Flow

1. Client sends `/v1/chat/completions` with `response_mode`
2. Chat router validates the request and passes `response_mode` into the initial RAG state
3. RAG prompt builder selects either the text profile or audio profile
4. LLM generates the answer under that profile
5. Output guard runs unchanged
6. Citation builder normalizes citations for both modes
7. Final answer formatting diverges:
   - Text mode: append `----` + `### Nguồn trích dẫn`
   - Audio mode: return the short spoken answer without appended citations

## Files Changed

| File | Action |
|------|--------|
| `api/orchestrator/models/schemas.py` | Add `response_mode` field to `ChatRequest` |
| `api/orchestrator/routers/chat.py` | Propagate `response_mode` into request execution and initial RAG state |
| `api/orchestrator/pipeline/rag.py` | Add response-mode-aware prompt building and citation footer formatting |
| `tests/orchestrator/test_schemas.py` | Validate `response_mode` defaults and allowed values |
| `tests/orchestrator/test_chat_router.py` | Verify router passes `response_mode` through correctly |
| `tests/orchestrator/test_rag_pipeline.py` | Verify text/audio prompt behavior and final answer formatting |

## Error Handling

- Unknown `response_mode` values should fail schema validation at the request boundary.
- If no citations are available, text mode should still return a clean structured answer without an empty `Nguồn trích dẫn` section.
- If the output guard forces a denial, the denial message should bypass the structured footer logic just as it does today unless safe output continues through citations.
- Audio mode should never append markdown-only citation sections that TTS would have to read aloud.

## Testing

- Schema tests for `response_mode="text"` default behavior and `"audio"` acceptance
- Router tests that the initial RAG state receives the expected `response_mode`
- RAG tests that text mode appends:
  - `----`
  - `### Nguồn trích dẫn`
- RAG tests that audio mode preserves `citations` in the result but does not append the footer to `answer`
- Prompt-construction tests that:
  - text mode contains structure instructions for `###` and `----`
  - audio mode contains short spoken-style instructions
- Regression coverage to ensure streaming returns the same final answer body that non-streaming returns
