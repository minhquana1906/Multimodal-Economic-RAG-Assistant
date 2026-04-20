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
from orchestrator.pipeline.rag_context import build_citation_section
from orchestrator.pipeline.rag_prompts import build_generation_prompt, resolve_rag_system_prompt
from orchestrator.services.conversation import extract_latest_user_query, normalize_messages


def _build_initial_state(
    *,
    raw_query: str,
    resolved_query: str,
) -> RAGState:
    return {
        "query": raw_query,
        "raw_query": raw_query,
        "resolved_query": resolved_query,
        "task_type": "rag",
        "embeddings": [],
        "retrieved_docs": [],
        "reranked_docs": [],
        "web_results": [],
        "final_context": [],
        "answer": "",
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
        )
    )
    return {
        "answer": result.get("answer", ""),
        "citations": result.get("citations", []),
        "task_type": "rag",
        "resolved_query": resolved_query,
    }


def create_chat_router(
    rag_graph,
    retrieval_graph=None,
    task_llm=None,
    prompts: PromptsConfig | None = None,
    settings=None,
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

            if request.stream and retrieval_graph is not None:
                return await _stream_response(request)

            # Non-streaming path
            result = await execute_chat_turn(
                rag_graph,
                task_llm,
                request.messages,
                request.max_tokens,
                prompts=prompts,
            )
            answer: str = result.get("answer", "")
            return ChatResponse(
                model=request.model,
                choices=[ChatCompletionChoice(message=AssistantMessage(content=answer))],
            )

    async def _stream_response(request: ChatRequest):
        normalized = normalize_messages(request.messages)
        raw_query = extract_latest_user_query(normalized)
        serialized = [{"role": m.role, "content": m.text_content()} for m in normalized]

        # Detect intent (direct chat vs RAG)
        intent = {"route": "rag", "resolved_query": raw_query}
        if task_llm is not None and hasattr(task_llm, "detect_intent"):
            intent = await task_llm.detect_intent(serialized)

        route = intent.get("route", "rag")
        resolved_query = intent.get("resolved_query") or raw_query

        if route == "direct":
            stream_messages = [
                {"role": "system", "content": prompts.direct_system_prompt},
                {"role": "user", "content": resolved_query},
            ]
            citation_suffix = ""
        else:
            # Run retrieval only — no LLM call, so first token starts after ~2-3s
            retrieval_state = await retrieval_graph.ainvoke(
                _build_initial_state(raw_query=raw_query, resolved_query=resolved_query)
            )
            final_context = retrieval_state.get("final_context", [])

            if settings is not None:
                user_prompt = build_generation_prompt(retrieval_state, settings)
                citation_suffix = (
                    build_citation_section(final_context, settings.rag.context_limit)
                    if final_context
                    else ""
                )
            else:
                user_prompt = resolved_query
                citation_suffix = ""

            system_prompt = resolve_rag_system_prompt(prompts)
            stream_messages = [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ]

        async def generate():
            if task_llm is not None and hasattr(task_llm, "stream_chat"):
                is_first = True
                async for delta in task_llm.stream_chat(stream_messages):
                    role = "assistant" if is_first else None
                    is_first = False
                    chunk = ChatStreamChunk(
                        model=request.model,
                        choices=[ChatStreamChoice(delta=ChatDelta(role=role, content=delta))],
                    )
                    yield f"data: {chunk.model_dump_json(exclude_none=True)}\n\n"

                if citation_suffix:
                    cite_chunk = ChatStreamChunk(
                        model=request.model,
                        choices=[ChatStreamChoice(delta=ChatDelta(content=citation_suffix))],
                    )
                    yield f"data: {cite_chunk.model_dump_json(exclude_none=True)}\n\n"
            else:
                # No streaming support — send empty stop chunk
                pass

            stop_chunk = ChatStreamChunk(
                model=request.model,
                choices=[ChatStreamChoice(delta=ChatDelta(), finish_reason="stop")],
            )
            yield f"data: {stop_chunk.model_dump_json(exclude_none=True)}\n\n"
            yield "data: [DONE]\n\n"

        return StreamingResponse(
            generate(),
            media_type="text/event-stream",
            headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
        )

    return router
