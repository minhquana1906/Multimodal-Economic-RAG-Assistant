from __future__ import annotations

import json
import logging

from fastapi import APIRouter, HTTPException
from fastapi.responses import StreamingResponse

from orchestrator.models.schemas import (
    ChatRequest,
    ChatResponse,
    ChatChoice,
    ChatDelta,
)
from orchestrator.pipeline.rag import RAGState

logger = logging.getLogger(__name__)

router = APIRouter()


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
    """Return an APIRouter with /v1/chat/completions bound to the given RAG graph."""

    @router.post("/v1/chat/completions")
    async def chat_completions(request: ChatRequest):
        # Extract the last user message
        user_message = next(
            (msg.content for msg in reversed(request.messages) if msg.role == "user"),
            None,
        )
        if not user_message:
            raise HTTPException(status_code=400, detail="No user message found")

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
            for i, word in enumerate(words):
                chunk = ChatChoice(
                    delta=ChatDelta(content=word + (" " if i < len(words) - 1 else "")),
                    finish_reason=None,
                )
                yield f"data: {json.dumps(chunk.model_dump())}\n\n"

            if citations:
                citations_text = "\n".join(
                    f"- {c.get('title', '')} ({c.get('source', '')})"
                    for c in citations
                )
                cite_chunk = ChatChoice(
                    delta=ChatDelta(content=f"\n\n**Nguồn:**\n{citations_text}"),
                    finish_reason="stop",
                )
                yield f"data: {json.dumps(cite_chunk.model_dump())}\n\n"

            yield "data: [DONE]\n\n"

        return StreamingResponse(generate(), media_type="text/event-stream")

    return router
