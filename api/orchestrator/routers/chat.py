from __future__ import annotations

import uuid
from typing import Any

from fastapi import APIRouter, HTTPException
from fastapi.responses import StreamingResponse
from loguru import logger

from orchestrator.models.schemas import (
    ChatChoice,
    ChatDelta,
    ChatRequest,
    ChatResponse,
    ChatStreamChunk,
)
from orchestrator.pipeline.rag import RAGState

_AUXILIARY_TASK_MARKERS = (
    "generate a concise, 3-5 word title",
    "generate a concise 3-5 word title",
    "generate 1-3 broad tags categorizing the main themes",
    "suggest 3-5 relevant follow-up questions",
)


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
        "input_guard_result": {},
        "output_guard_result": {},
        "generation_prompt": "",
        "citations": [],
        "error": None,
    }


def _is_auxiliary_task_prompt(prompt: str) -> bool:
    normalized = " ".join(prompt.lower().split())
    return normalized.startswith("### task:") and any(
        marker in normalized for marker in _AUXILIARY_TASK_MARKERS
    )


async def _run_request(rag_graph: Any, task_llm: Any | None, user_message: str) -> dict:
    if task_llm is not None and _is_auxiliary_task_prompt(user_message):
        return {
            "answer": await task_llm.complete_prompt(user_message),
            "citations": [],
        }
    return await rag_graph.ainvoke(_build_initial_state(user_message))


def _format_citation(citation: dict) -> str:
    title = citation.get("title", "") or "Nguồn tham khảo"
    url = citation.get("url", "") or ""
    source = citation.get("source", "") or url or "unknown"
    score = float(citation.get("score", 0.0) or 0.0)
    title_part = f"[{title}]({url})" if url else title
    return f"- {title_part} - **{source} ({score:.4f})**"


def create_chat_router(rag_graph, task_llm=None) -> APIRouter:
    """Return a fresh APIRouter with /v1/chat/completions bound to the given RAG graph."""
    router = APIRouter()

    @router.post("/v1/chat/completions")
    async def chat_completions(request: ChatRequest):
        user_message = next(
            (msg.content for msg in reversed(request.messages) if msg.role == "user"),
            None,
        )
        if not user_message:
            raise HTTPException(status_code=400, detail="No user message found")

        request_id = str(uuid.uuid4())[:8]
        with logger.contextualize(request_id=request_id):
            result = await _run_request(rag_graph, task_llm, user_message)

            answer: str = result.get("answer", "")
            citations: list[dict] = result.get("citations", [])

            if not request.stream:
                return ChatResponse(
                    model=request.model,
                    choices=[
                        ChatChoice(delta=ChatDelta(role="assistant", content=answer))
                    ],
                )

            async def generate():
                words = answer.split()
                if words:
                    first_chunk = ChatStreamChunk(
                        model=request.model,
                        choices=[
                            ChatChoice(
                                delta=ChatDelta(
                                    role="assistant",
                                    content=words[0] + (" " if len(words) > 1 else ""),
                                ),
                                finish_reason=None,
                            )
                        ],
                    )
                    yield f"data: {first_chunk.model_dump_json(exclude_none=True)}\n\n"

                    for i, word in enumerate(words[1:], start=1):
                        is_last = (i == len(words) - 1) and not citations
                        chunk = ChatStreamChunk(
                            model=request.model,
                            choices=[
                                ChatChoice(
                                    delta=ChatDelta(
                                        content=word + (" " if i < len(words) - 1 else "")
                                    ),
                                    finish_reason="stop" if is_last else None,
                                )
                            ],
                        )
                        yield f"data: {chunk.model_dump_json(exclude_none=True)}\n\n"
                else:
                    stop_chunk = ChatStreamChunk(
                        model=request.model,
                        choices=[ChatChoice(delta=ChatDelta(), finish_reason="stop")],
                    )
                    yield f"data: {stop_chunk.model_dump_json(exclude_none=True)}\n\n"

                if citations:
                    citations_text = "\n".join(_format_citation(c) for c in citations)
                    cite_chunk = ChatStreamChunk(
                        model=request.model,
                        choices=[
                            ChatChoice(
                                delta=ChatDelta(
                                    content=f"\n\n**Nguồn:**\n{citations_text}"
                                ),
                                finish_reason="stop",
                            )
                        ],
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
