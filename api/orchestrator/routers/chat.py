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
from orchestrator.pipeline.rag import RAGState, _build_denial_message
from orchestrator.services.conversation import (
    DEFAULT_PROMPT_TOKEN_BUDGET,
    DEFAULT_RECENT_TURNS,
    build_auxiliary_history,
    build_auxiliary_prompt,
    classify_chat_route,
    build_conversation_context,
    classify_task,
    extract_latest_user_query,
    is_live_facts_general_chat_query,
    normalize_messages,
    resolve_user_query,
    should_summarize,
    summarize_history,
)


_AUXILIARY_TASKS = {"title", "tags", "follow_ups"}


def _build_initial_state(
    *,
    raw_query: str,
    resolved_query: str,
    conversation_summary: str,
    conversation_context: str,
    task_type: str,
    response_mode: str,
) -> RAGState:
    return {
        "query": raw_query,
        "raw_query": raw_query,
        "resolved_query": resolved_query,
        "conversation_summary": conversation_summary,
        "conversation_context": conversation_context,
        "task_type": task_type,
        "response_mode": response_mode,
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
        "citation_pool": {},
        "error": None,
    }


async def _prepare_conversation(
    messages,
    max_tokens: int | None,
    query_llm: Any | None,
    prompts: PromptsConfig,
) -> dict:
    normalized_messages = normalize_messages(messages)
    raw_query = extract_latest_user_query(normalized_messages)
    task_type = classify_task(normalized_messages, raw_query)
    non_system_messages = [message for message in normalized_messages if message.role != "system"]
    recent_turns = non_system_messages[-DEFAULT_RECENT_TURNS:]
    prior_turns = non_system_messages[:-1]

    prompt_token_budget = max_tokens or DEFAULT_PROMPT_TOKEN_BUDGET
    conversation_summary = ""
    if should_summarize(
        normalized_messages,
        prompt_token_budget=prompt_token_budget,
    ):
        conversation_summary = summarize_history(normalized_messages)

    conversation_context = build_conversation_context(
        conversation_summary,
        recent_turns,
    )
    resolved_query = raw_query
    if task_type not in _AUXILIARY_TASKS:
        resolved_query = await resolve_user_query(
            raw_query=raw_query,
            summary=conversation_summary,
            recent_turns=prior_turns[-DEFAULT_RECENT_TURNS:],
            llm=query_llm,
            prompt_instruction=prompts.query_rewrite_prompt,
            max_tokens=max_tokens,
        )
        task_type = await classify_chat_route(
            resolved_query=resolved_query,
            conversation_context=conversation_context,
            llm=query_llm,
            classifier_prompt=prompts.route_classifier_prompt,
        )

    return {
        "messages": normalized_messages,
        "raw_query": raw_query,
        "resolved_query": resolved_query,
        "conversation_summary": conversation_summary,
        "conversation_context": conversation_context,
        "task_type": task_type,
    }


def _resolve_response_mode(request: ChatRequest) -> str:
    if "response_mode" in request.model_fields_set:
        return request.response_mode

    if request.audio or any(
        modality.lower() == "audio" for modality in (request.modalities or [])
    ):
        return "audio"

    return request.response_mode


def _general_chat_response_contract(prompts: PromptsConfig, response_mode: str) -> str:
    if response_mode == "audio":
        return prompts.general_chat_audio_response_contract
    return prompts.general_chat_text_response_contract


def _build_general_chat_prompt(
    *,
    prompts: PromptsConfig,
    resolved_query: str,
    conversation_context: str,
    response_mode: str,
) -> str:
    context_block = conversation_context.strip() or "Ngữ cảnh hội thoại:\nKhông có."
    return prompts.general_chat_user_template.format(
        conversation_context=context_block,
        response_contract=_general_chat_response_contract(prompts, response_mode),
        question=resolved_query.strip(),
    )


def _build_general_chat_retry_prompt(
    *,
    prompts: PromptsConfig,
    original_prompt: str,
    unsafe_answer: str,
    guard_result: dict,
) -> str:
    categories = ", ".join(guard_result.get("categories", [])) or "Unknown"
    refusal = guard_result.get("refusal") or "Unknown"
    safe_label = guard_result.get("safe_label") or "Unsafe"
    return prompts.general_chat_retry_prompt.format(
        original_prompt=original_prompt,
        unsafe_answer=unsafe_answer,
        safe_label=safe_label,
        categories=categories,
        refusal=refusal,
    )


async def _handle_general_chat(
    *,
    llm: Any,
    guard: Any,
    prompts: PromptsConfig,
    conversation: dict,
    response_mode: str,
) -> dict:
    guard_result = await guard.check_input(conversation["resolved_query"])
    if guard_result["label"] != "safe":
        return {
            "answer": _build_denial_message(
                prompts.apology_message,
                guard_result.get("categories", []),
            ),
            "citations": [],
            "task_type": conversation["task_type"],
            "resolved_query": conversation["resolved_query"],
        }

    if is_live_facts_general_chat_query(conversation["resolved_query"]):
        answer = prompts.general_chat_live_facts_message
        output_guard_result = await guard.check_output(
            text=answer,
            prompt=conversation["resolved_query"],
        )
        if output_guard_result["label"] != "safe":
            answer = _build_denial_message(
                prompts.guard_error_message,
                output_guard_result.get("categories", []),
            )
        return {
            "answer": answer,
            "citations": [],
            "task_type": conversation["task_type"],
            "resolved_query": conversation["resolved_query"],
        }

    user_prompt = _build_general_chat_prompt(
        prompts=prompts,
        resolved_query=conversation["resolved_query"],
        conversation_context=conversation["conversation_context"],
        response_mode=response_mode,
    )
    answer = await llm.generate(
        system_prompt=prompts.general_chat_system_prompt,
        user_prompt=user_prompt,
    )
    output_guard_result = await guard.check_output(
        text=answer,
        prompt=conversation["resolved_query"],
    )
    if output_guard_result["label"] == "safe":
        return {
            "answer": answer,
            "citations": [],
            "task_type": conversation["task_type"],
            "resolved_query": conversation["resolved_query"],
        }

    retry_prompt = _build_general_chat_retry_prompt(
        prompts=prompts,
        original_prompt=user_prompt,
        unsafe_answer=answer,
        guard_result=output_guard_result,
    )
    retry_answer = await llm.generate(
        system_prompt=prompts.general_chat_system_prompt,
        user_prompt=retry_prompt,
    )
    retry_guard = await guard.check_output(
        text=retry_answer,
        prompt=conversation["resolved_query"],
    )
    if retry_guard["label"] == "safe":
        answer = retry_answer
    else:
        answer = _build_denial_message(
            prompts.guard_error_message,
            retry_guard.get("categories", []),
        )
    return {
        "answer": answer,
        "citations": [],
        "task_type": conversation["task_type"],
        "resolved_query": conversation["resolved_query"],
    }


async def _run_request(
    rag_graph: Any,
    task_llm: Any | None,
    guard: Any | None,
    prompts: PromptsConfig,
    messages,
    max_tokens: int | None,
    response_mode: str = "text",
) -> dict:
    conversation = await _prepare_conversation(messages, max_tokens, task_llm, prompts)
    if task_llm is not None and conversation["task_type"] in _AUXILIARY_TASKS:
        history = build_auxiliary_history(
            conversation["messages"],
            conversation["raw_query"],
        )
        prompt = build_auxiliary_prompt(conversation["task_type"], history)
        return {
            "answer": await task_llm.complete_prompt(prompt),
            "citations": [],
            "task_type": conversation["task_type"],
            "resolved_query": conversation["resolved_query"],
        }
    if (
        task_llm is not None
        and guard is not None
        and conversation["task_type"] == "general_chat"
    ):
        return await _handle_general_chat(
            llm=task_llm,
            guard=guard,
            prompts=prompts,
            conversation=conversation,
            response_mode=response_mode,
        )

    result = await rag_graph.ainvoke(
        _build_initial_state(
            raw_query=conversation["raw_query"],
            resolved_query=conversation["resolved_query"],
            conversation_summary=conversation["conversation_summary"],
            conversation_context=conversation["conversation_context"],
            task_type=conversation["task_type"],
            response_mode=response_mode,
        )
    )
    return {
        **result,
        "task_type": conversation["task_type"],
        "resolved_query": conversation["resolved_query"],
    }


@traceable(name="Execute Chat Turn")
async def execute_chat_turn(
    rag_graph: Any,
    task_llm: Any | None,
    messages,
    max_tokens: int | None,
    response_mode: str = "text",
    guard: Any | None = None,
    prompts: PromptsConfig | None = None,
) -> dict:
    prompts = prompts or PromptsConfig()
    result = await _run_request(
        rag_graph,
        task_llm,
        guard,
        prompts,
        messages,
        max_tokens,
        response_mode=response_mode,
    )
    return {
        "answer": result.get("answer", ""),
        "citations": result.get("citations", []),
        "task_type": result.get("task_type", "rag"),
        "resolved_query": result.get("resolved_query", ""),
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
        effective_response_mode = _resolve_response_mode(request)
        with logger.contextualize(request_id=request_id):
            logger.info(
                "chat request: requested_response_mode={} effective_response_mode={} "
                "modalities={} has_audio={}",
                request.response_mode,
                effective_response_mode,
                request.modalities or [],
                bool(request.audio),
            )
            result = await execute_chat_turn(
                rag_graph,
                task_llm,
                request.messages,
                request.max_tokens,
                effective_response_mode,
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
