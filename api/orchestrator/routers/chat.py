from __future__ import annotations

import uuid

from fastapi import APIRouter, HTTPException
from fastapi.responses import StreamingResponse
from loguru import logger

from orchestrator.models.schemas import (
    ChatRequest,
    ChatResponse,
    ChatChoice,
    ChatDelta,
    ChatStreamChunk,
)
from orchestrator.pipeline.rag import RAGState


def _build_initial_state(query: str) -> RAGState:
    return {
        "query": query,
        "input_safe": False,
        "embeddings": [],
        "retrieved_docs": [],
        "reranked_docs": [],
        "web_results": [],
        "final_context": [],
        "answer": "",
        "output_safe": False,
        "citations": [],
        "error": None,
    }


def create_chat_router(rag_graph) -> APIRouter:
    """Return a fresh APIRouter with /v1/chat/completions bound to the given RAG graph."""
    router = APIRouter()   # ← fresh per call, not a module-level singleton

    @router.post("/v1/chat/completions")
    async def chat_completions(request: ChatRequest):
        # Extract the last user message
        user_message = next(
            (msg.content for msg in reversed(request.messages) if msg.role == "user"),
            None,
        )
        if not user_message:
            raise HTTPException(status_code=400, detail="No user message found")

        request_id = str(uuid.uuid4())[:8]
        with logger.contextualize(request_id=request_id):
            result = await rag_graph.ainvoke(_build_initial_state(user_message))

            answer: str = result.get("answer", "")
            citations: list[dict] = result.get("citations", [])

            if not request.stream:
                return ChatResponse(
                    model=request.model,
                    choices=[ChatChoice(delta=ChatDelta(role="assistant", content=answer))],
                )

            # SSE streaming: answer word-by-word, then citations block, then [DONE]
            async def generate():
                words = answer.split()
                # First chunk: include role
                if words:
                    first_chunk = ChatStreamChunk(
                        model=request.model,
                        choices=[ChatChoice(delta=ChatDelta(role="assistant", content=words[0] + (" " if len(words) > 1 else "")), finish_reason=None)],
                    )
                    yield f"data: {first_chunk.model_dump_json(exclude_none=True)}\n\n"

                    # Subsequent word chunks: no role
                    for i, word in enumerate(words[1:], start=1):
                        is_last = (i == len(words) - 1) and not citations
                        chunk = ChatStreamChunk(
                            model=request.model,
                            choices=[ChatChoice(
                                delta=ChatDelta(content=word + (" " if i < len(words) - 1 else "")),
                                finish_reason="stop" if is_last else None,
                            )],
                        )
                        yield f"data: {chunk.model_dump_json(exclude_none=True)}\n\n"
                else:
                    # Empty answer — emit a stop chunk immediately
                    stop_chunk = ChatStreamChunk(
                        model=request.model,
                        choices=[ChatChoice(delta=ChatDelta(), finish_reason="stop")],
                    )
                    yield f"data: {stop_chunk.model_dump_json(exclude_none=True)}\n\n"

                if citations:
                    citations_text = "\n".join(
                        f"- {c.get('title', '')} ({c.get('source', '')})"
                        for c in citations
                    )
                    cite_chunk = ChatStreamChunk(
                        model=request.model,
                        choices=[ChatChoice(
                            delta=ChatDelta(content=f"\n\n**Nguồn:**\n{citations_text}"),
                            finish_reason="stop",
                        )],
                    )
                    yield f"data: {cite_chunk.model_dump_json(exclude_none=True)}\n\n"

                yield "data: [DONE]\n\n"

            return StreamingResponse(
                generate(),
                media_type="text/event-stream",
                headers={
                    "Cache-Control": "no-cache",
                    "X-Accel-Buffering": "no",
                },
            )

    return router
