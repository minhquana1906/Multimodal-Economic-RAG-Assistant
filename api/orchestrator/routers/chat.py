from __future__ import annotations

import uuid
from typing import Any

from fastapi import APIRouter, HTTPException
from fastapi.responses import StreamingResponse
from langsmith import traceable
from loguru import logger

from orchestrator.config import PromptsConfig
from orchestrator.models.schemas import (
    AssistantMessage,
    ChatCompletionChoice,
    ChatDelta,
    ChatRequest,
    ChatResponse,
    ChatStreamChoice,
    ChatStreamChunk,
)
from orchestrator.pipeline.rag import RAGState
from orchestrator.services.conversation import extract_latest_user_query, normalize_messages


def _build_initial_state(
    *,
    raw_query: str,
    resolved_query: str,
    response_mode: str,
) -> RAGState:
    return {
        "query": raw_query,
        "raw_query": raw_query,
        "resolved_query": resolved_query,
        "conversation_summary": "",
        "conversation_context": "",
        "task_type": "rag",
        "response_mode": response_mode,
        "input_safe": True,
        "embeddings": [],
        "retrieved_docs": [],
        "reranked_docs": [],
        "web_results": [],
        "final_context": [],
        "answer": "",
        "output_safe": True,
        "input_guard_result": {},
        "output_guard_result": {},
        "generation_prompt": "",
        "citations": [],
        "citation_pool": {},
        "error": None,
    }


@traceable(name="Execute Chat Turn")
async def execute_chat_turn(
    rag_graph: Any,
    llm: Any | None,
    messages,
    max_tokens: int | None,
    response_mode: str = "text",
    guard: Any | None = None,
    prompts: PromptsConfig | None = None,
) -> dict:
    """Route request via detect_intent: direct skips the graph, rag invokes it."""
    prompts = prompts or PromptsConfig()
    normalized = normalize_messages(messages)
    raw_query = extract_latest_user_query(normalized)

    if not raw_query:
        return {"answer": "", "citations": [], "task_type": "rag", "resolved_query": ""}

    # Serialize messages for detect_intent
    serialized = [{"role": m.role, "content": m.text_content()} for m in normalized]

    intent = {"route": "rag", "resolved_query": raw_query}
    if llm is not None and hasattr(llm, "detect_intent"):
        intent = await llm.detect_intent(serialized)

    route = intent.get("route", "rag")
    resolved_query = intent.get("resolved_query") or raw_query

    if route == "direct" and llm is not None:
        answer = await llm.generate(
            system_prompt=prompts.direct_system_prompt,
            user_prompt=resolved_query,
        )
        return {
            "answer": answer,
            "citations": [],
            "task_type": "direct",
            "resolved_query": resolved_query,
        }

    # rag route
    result = await rag_graph.ainvoke(
        _build_initial_state(
            raw_query=raw_query,
            resolved_query=resolved_query,
            response_mode=response_mode,
        )
    )
    return {
        "answer": result.get("answer", ""),
        "citations": result.get("citations", []),
        "task_type": "rag",
        "resolved_query": resolved_query,
    }


def _chunk_answer(answer: str, chunk_size: int = 64) -> list[str]:
    if not answer:
        return []
    return [answer[index : index + chunk_size] for index in range(0, len(answer), chunk_size)]


def create_chat_router(
    rag_graph,
    task_llm=None,
    guard=None,
    prompts: PromptsConfig | None = None,
) -> APIRouter:
    """Return a fresh APIRouter with /v1/chat/completions bound to the given RAG graph."""
    router = APIRouter()
    prompts = prompts or PromptsConfig()

    @router.post("/v1/chat/completions")
    async def chat_completions(request: ChatRequest):
        user_message = extract_latest_user_query(request.messages)
        if not user_message:
            raise HTTPException(status_code=400, detail="No user message found")

        request_id = str(uuid.uuid4())[:8]
        with logger.contextualize(request_id=request_id):
            logger.info("chat request received")
            result = await execute_chat_turn(
                rag_graph,
                task_llm,
                request.messages,
                request.max_tokens,
                "text",
                guard=guard,
                prompts=prompts,
            )

            answer: str = result.get("answer", "")
            citations: list[dict] = result.get("citations", [])

            if not request.stream:
                return ChatResponse(
                    model=request.model,
                    choices=[
                        ChatCompletionChoice(
                            message=AssistantMessage(content=answer),
                        )
                    ],
                )

            async def generate():
                chunks = _chunk_answer(answer)
                if chunks:
                    first_chunk = ChatStreamChunk(
                        model=request.model,
                        choices=[
                            ChatStreamChoice(
                                delta=ChatDelta(
                                    role="assistant",
                                    content=chunks[0],
                                ),
                                finish_reason="stop" if len(chunks) == 1 else None,
                            )
                        ],
                    )
                    yield f"data: {first_chunk.model_dump_json(exclude_none=True)}\n\n"

                    for index, chunk_text in enumerate(chunks[1:], start=1):
                        is_last = index == len(chunks) - 1
                        chunk = ChatStreamChunk(
                            model=request.model,
                            choices=[
                                ChatStreamChoice(
                                    delta=ChatDelta(content=chunk_text),
                                    finish_reason="stop" if is_last else None,
                                )
                            ],
                        )
                        yield f"data: {chunk.model_dump_json(exclude_none=True)}\n\n"
                else:
                    stop_chunk = ChatStreamChunk(
                        model=request.model,
                        choices=[ChatStreamChoice(delta=ChatDelta(), finish_reason="stop")],
                    )
                    yield f"data: {stop_chunk.model_dump_json(exclude_none=True)}\n\n"

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
