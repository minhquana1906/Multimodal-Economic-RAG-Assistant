from __future__ import annotations

import time
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
from orchestrator.pipeline.rag_context import (
    build_citation_section,
    build_context_item,
    finalize_citations,
)
from orchestrator.pipeline.rag_policy import should_use_web_search_for_direct
from orchestrator.pipeline.rag_prompts import (
    build_direct_prompt,
    build_direct_prompt_with_web,
    build_generation_prompt,
    build_intent_prompt,
    resolve_rag_system_prompt,
)
from orchestrator.services import vision as _vision
from orchestrator.services.conversation import (
    extract_latest_user_images,
    extract_latest_user_query,
    normalize_messages,
)
from orchestrator.tracing import log_phase


def _build_initial_state(
    *,
    raw_query: str,
    resolved_query: str,
    image_parts: list | None = None,
    image_caption: str = "",
) -> RAGState:
    return {
        "query": raw_query,
        "raw_query": raw_query,
        "resolved_query": resolved_query,
        "task_type": "rag",
        "image_parts": image_parts or [],
        "image_caption": image_caption,
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


async def _preflight_image(
    *,
    llm,
    image_parts: list,
    raw_query: str,
    prompts: "PromptsConfig",
    vision_cfg,
) -> tuple[list[dict], str, str]:
    """Resize/validate images and get caption + rag_query via MLLM.

    Returns (processed_image_dicts, caption, rag_query).
    Returns ([], "", raw_query) when no images or vision is disabled.
    """
    if not image_parts or llm is None:
        return [], "", raw_query
    if vision_cfg is not None and not getattr(vision_cfg, "enable_vision", True):
        return [], "", raw_query

    max_imgs = getattr(vision_cfg, "max_images_per_turn", 4) if vision_cfg else 4
    max_pixels = getattr(vision_cfg, "max_image_pixels", 1_048_576) if vision_cfg else 1_048_576
    max_bytes = getattr(vision_cfg, "max_image_bytes", 4_000_000) if vision_cfg else 4_000_000

    capped = image_parts[:max_imgs]
    processed = [
        await _vision.process_image_part(p, max_pixels=max_pixels, max_bytes=max_bytes)
        for p in capped
    ]

    system_prompt = getattr(prompts, "image_caption_system_prompt", "")
    user_template = getattr(
        prompts,
        "image_caption_user_template",
        'Phân tích ảnh. Trả JSON: {{"caption": "...", "rag_query": "..."}}. Yêu cầu: {user_text}',
    )
    desc = await llm.describe_image(
        user_text=raw_query,
        image_content_parts=processed,
        system_prompt=system_prompt,
        user_template=user_template,
    )
    return processed, desc.get("caption", ""), desc.get("rag_query", "") or raw_query


@traceable(name="Execute Chat Turn")
async def execute_chat_turn(
    rag_graph: Any,
    llm: Any | None,
    messages,
    max_tokens: int | None,
    prompts: PromptsConfig | None = None,
    web_search: Any | None = None,
    settings: Any | None = None,
) -> dict:
    """Route request via detect_intent: direct skips the graph, rag invokes it."""
    prompts = prompts or PromptsConfig()
    normalized = normalize_messages(messages)
    raw_query = extract_latest_user_query(normalized)
    image_parts_raw = extract_latest_user_images(normalized)

    # Allow image-only messages
    if not raw_query and image_parts_raw:
        raw_query = "Phân tích nội dung ảnh này"

    if not raw_query and not image_parts_raw:
        return {"answer": "", "citations": [], "task_type": "rag", "resolved_query": ""}

    # Vision preflight: resize images + get caption/rag_query
    vision_cfg = getattr(settings, "llm", None) if settings else None
    processed_images, caption, rag_query_from_image = await _preflight_image(
        llm=llm,
        image_parts=image_parts_raw,
        raw_query=raw_query,
        prompts=prompts,
        vision_cfg=vision_cfg,
    )

    intent = {"route": "rag", "resolved_query": rag_query_from_image or raw_query}
    if llm is not None and hasattr(llm, "detect_intent"):
        intent_system_prompt, intent_user_prompt = build_intent_prompt(
            normalized, prompts, image_caption=caption
        )
        intent = await llm.detect_intent(
            system_prompt=intent_system_prompt,
            user_prompt=intent_user_prompt,
            fallback_query=rag_query_from_image or raw_query,
        )

    route = intent.get("route", "rag")
    resolved_query = intent.get("resolved_query") or rag_query_from_image or raw_query
    logger.info(f"route_decided route={route} resolved_query_len={len(resolved_query)} has_images={bool(processed_images)}")

    if route == "direct" and llm is not None:
        use_web, web_reason = (False, "disabled")
        if web_search is not None and settings is not None:
            use_web, web_reason = should_use_web_search_for_direct(resolved_query, settings)

        if use_web:
            logger.info(f"direct_web_search=true reason={web_reason}")
            async with log_phase("direct_web") as ctx:
                web_results = await web_search.search(resolved_query)
                ctx["hits"] = len(web_results)

            if web_results:
                web_context = [
                    build_context_item(
                        item,
                        source_type="web",
                        retrieval_stage="web_fallback",
                        original_rank=i,
                        context_id=item.get("context_id", f"web:{i}"),
                    )
                    for i, item in enumerate(web_results)
                ]
                user_prompt = build_direct_prompt_with_web(
                    messages=normalized,
                    resolved_query=resolved_query,
                    web_results=web_results,
                    prompts=prompts,
                )
                system_prompt = resolve_rag_system_prompt(prompts)
                answer = await llm.generate(
                    system_prompt=system_prompt,
                    user_prompt=user_prompt,
                )
                context_limit = getattr(getattr(settings, "rag", None), "context_limit", 10)
                pseudo_state = {
                    "answer": answer,
                    "final_context": web_context,
                    "citation_pool": {item["context_id"]: item for item in web_context},
                    "response_mode": "text",
                }
                result = finalize_citations(
                    pseudo_state,
                    context_limit=context_limit,
                    citation_limit=context_limit,
                )
                return {
                    "answer": result["answer"],
                    "citations": result["citations"],
                    "task_type": "direct_web",
                    "resolved_query": resolved_query,
                }

        logger.info(f"direct_web_search=false reason={web_reason}")
        direct_user_prompt = build_direct_prompt(
            messages=normalized,
            resolved_query=resolved_query,
            prompts=prompts,
        )
        if processed_images and hasattr(llm, "generate_with_images"):
            answer = await llm.generate_with_images(
                system_prompt=prompts.direct_system_prompt,
                user_text=direct_user_prompt,
                image_content_parts=processed_images,
            )
        else:
            answer = await llm.generate(
                system_prompt=prompts.direct_system_prompt,
                user_prompt=direct_user_prompt,
            )
        return {
            "answer": answer,
            "citations": [],
            "task_type": "direct",
            "resolved_query": resolved_query,
        }

    # rag route
    async with log_phase("rag_graph") as ctx:
        result = await rag_graph.ainvoke(
            _build_initial_state(
                raw_query=raw_query,
                resolved_query=resolved_query,
                image_parts=processed_images,
                image_caption=caption,
            )
        )
        ctx["citations"] = len(result.get("citations", []))
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
    web_search=None,
) -> APIRouter:
    """Return a fresh APIRouter with /v1/chat/completions bound to the given RAG graph."""
    router = APIRouter()
    prompts = prompts or PromptsConfig()

    @router.post("/v1/chat/completions")
    async def chat_completions(request: ChatRequest):
        normalized_msgs = normalize_messages(request.messages)
        user_message = extract_latest_user_query(normalized_msgs)
        user_images = extract_latest_user_images(normalized_msgs)
        if not user_message and not user_images:
            raise HTTPException(status_code=400, detail="No user message found")

        request_id = str(uuid.uuid4())[:8]
        t_start = time.monotonic()
        with logger.contextualize(request_id=request_id):
            logger.info(
                f"request_received stream={request.stream} model={request.model} query_len={len(user_message)}"
            )

            if request.stream and retrieval_graph is not None:
                return _stream_response(request, t_start)

            # Non-streaming path
            result = await execute_chat_turn(
                rag_graph,
                task_llm,
                request.messages,
                request.max_tokens,
                prompts=prompts,
                web_search=web_search,
                settings=settings,
            )
            answer: str = result.get("answer", "")
            total_ms = int((time.monotonic() - t_start) * 1000)
            logger.info(
                f"request_completed route={result.get('task_type', 'rag')} total_ms={total_ms}"
            )
            return ChatResponse(
                model=request.model,
                choices=[ChatCompletionChoice(message=AssistantMessage(content=answer))],
            )

    def _stream_response(request: ChatRequest, t_start: float):
        normalized = normalize_messages(request.messages)
        raw_query = extract_latest_user_query(normalized)
        image_parts_raw = extract_latest_user_images(normalized)
        if not raw_query and image_parts_raw:
            raw_query = "Phân tích nội dung ảnh này"
        route = "rag"

        async def generate():
            nonlocal route

            # Wrap the entire streaming flow under one LangSmith parent span.
            try:
                from langsmith import trace as _ls_trace
                ls_ctx = _ls_trace(
                    "Stream Chat Turn",
                    run_type="chain",
                    inputs={"query": raw_query, "model": request.model},
                )
            except Exception:
                import contextlib
                ls_ctx = contextlib.nullcontext()

            stream_messages: list[dict] = []
            citation_suffix = ""

            async with ls_ctx:
                # Vision preflight
                vision_cfg = getattr(settings, "llm", None) if settings else None
                processed_images, caption, rag_query_from_image = await _preflight_image(
                    llm=task_llm,
                    image_parts=image_parts_raw,
                    raw_query=raw_query,
                    prompts=prompts,
                    vision_cfg=vision_cfg,
                )

                # Detect intent (text-only; caption injected)
                intent = {"route": "rag", "resolved_query": rag_query_from_image or raw_query}
                if task_llm is not None and hasattr(task_llm, "detect_intent"):
                    intent_s, intent_u = build_intent_prompt(
                        normalized, prompts, image_caption=caption
                    )
                    intent = await task_llm.detect_intent(
                        system_prompt=intent_s,
                        user_prompt=intent_u,
                        fallback_query=rag_query_from_image or raw_query,
                    )

                route = intent.get("route", "rag")
                resolved_query = intent.get("resolved_query") or rag_query_from_image or raw_query
                logger.info(f"route_decided route={route} resolved_query_len={len(resolved_query)} has_images={bool(processed_images)}")

                def _user_content(text: str) -> "str | list":
                    if not processed_images:
                        return text
                    return list(processed_images) + [{"type": "text", "text": text}]

                if route == "direct":
                    use_web, web_reason = (False, "disabled")
                    if web_search is not None and settings is not None:
                        use_web, web_reason = should_use_web_search_for_direct(
                            resolved_query, settings
                        )

                    if use_web:
                        logger.info(f"direct_web_search=true reason={web_reason}")
                        async with log_phase("direct_web") as ctx:
                            web_results = await web_search.search(resolved_query)
                            ctx["hits"] = len(web_results)

                        if web_results:
                            web_context = [
                                build_context_item(
                                    item,
                                    source_type="web",
                                    retrieval_stage="web_fallback",
                                    original_rank=i,
                                    context_id=item.get("context_id", f"web:{i}"),
                                )
                                for i, item in enumerate(web_results)
                            ]
                            user_prompt = build_direct_prompt_with_web(
                                messages=normalized,
                                resolved_query=resolved_query,
                                web_results=web_results,
                                prompts=prompts,
                            )
                            context_limit = getattr(
                                getattr(settings, "rag", None), "context_limit", 10
                            )
                            citation_suffix = build_citation_section(web_context, context_limit)
                            stream_messages = [
                                {"role": "system", "content": resolve_rag_system_prompt(prompts)},
                                {"role": "user", "content": _user_content(user_prompt)},
                            ]
                            route = "direct_web"
                        else:
                            use_web = False

                    if not use_web:
                        logger.info(f"direct_web_search=false reason={web_reason}")
                        direct_user_prompt = build_direct_prompt(
                            messages=normalized,
                            resolved_query=resolved_query,
                            prompts=prompts,
                        )
                        stream_messages = [
                            {"role": "system", "content": prompts.direct_system_prompt},
                            {"role": "user", "content": _user_content(direct_user_prompt)},
                        ]

                else:
                    # RAG: run retrieval graph, then stream generation outside graph
                    async with log_phase("retrieval_graph") as ctx:
                        retrieval_state = await retrieval_graph.ainvoke(
                            _build_initial_state(
                                raw_query=raw_query,
                                resolved_query=resolved_query,
                                image_parts=processed_images,
                                image_caption=caption,
                            )
                        )
                        ctx["web_results"] = len(retrieval_state.get("web_results", []))
                        ctx["context_items"] = len(retrieval_state.get("final_context", []))

                    final_context = retrieval_state.get("final_context", [])
                    if settings is not None:
                        user_prompt = build_generation_prompt(retrieval_state, settings)
                        context_limit = settings.rag.context_limit
                        citation_suffix = (
                            build_citation_section(final_context, context_limit)
                            if final_context
                            else ""
                        )
                    else:
                        user_prompt = resolved_query
                        citation_suffix = ""

                    system_prompt = resolve_rag_system_prompt(prompts)
                    stream_messages = [
                        {"role": "system", "content": system_prompt},
                        {"role": "user", "content": _user_content(user_prompt)},
                    ]

                # Stream LLM response — stream_chat is @traceable, becomes child span
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

            total_ms = int((time.monotonic() - t_start) * 1000)
            logger.info(f"request_completed route={route} total_ms={total_ms} stream=true")

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
