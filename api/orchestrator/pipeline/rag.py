from __future__ import annotations

import re
from typing import Literal, TypedDict

from langgraph.graph import END, START, StateGraph
from langsmith import traceable
from loguru import logger


CATEGORY_REASON_MAP = {
    "Violent": "nội dung liên quan đến bạo lực",
    "Non-violent Illegal Acts": "nội dung hướng dẫn hành vi trái phép",
    "Sexual Content or Sexual Acts": "nội dung tình dục không phù hợp",
    "PII": "nội dung yêu cầu hoặc tiết lộ thông tin cá nhân nhạy cảm",
    "Suicide & Self-Harm": "nội dung liên quan đến tự hại hoặc tự sát",
    "Unethical Acts": "nội dung mang tính phi đạo đức hoặc gây hại",
    "Politically Sensitive Topics": "nội dung nhạy cảm về chính trị",
    "Copyright Violation": "nội dung có nguy cơ vi phạm bản quyền",
    "Jailbreak": "yêu cầu tìm cách vượt qua giới hạn an toàn",
}
DEFAULT_GUARD_REASON = "nội dung vi phạm chính sách an toàn"
_CITATION_PATTERN = re.compile(r"\[\[cite:([a-zA-Z0-9:_-]+)\]\]")


class RAGState(TypedDict):
    """Complete RAG pipeline state passed between nodes."""

    query: str
    raw_query: str
    resolved_query: str
    conversation_summary: str
    conversation_context: str
    task_type: str
    input_safe: bool
    embeddings: list[float]
    retrieved_docs: list[dict]
    reranked_docs: list[dict]
    web_results: list[dict]
    final_context: list[dict]
    answer: str
    output_safe: bool
    input_guard_result: dict
    output_guard_result: dict
    generation_prompt: str
    citations: list[dict]
    citation_pool: dict[str, dict]
    error: str | None


def _raw_query(state: RAGState) -> str:
    return state.get("raw_query") or state.get("query", "")


def _resolved_query(state: RAGState) -> str:
    return state.get("resolved_query") or _raw_query(state)


def _primary_guard_reason(categories: list[str]) -> str:
    for category in categories:
        if category in CATEGORY_REASON_MAP:
            return CATEGORY_REASON_MAP[category]
    return DEFAULT_GUARD_REASON


def _build_denial_message(base_message: str, categories: list[str]) -> str:
    reason = _primary_guard_reason(categories)
    return f"{base_message} Lý do: {reason}."


def _normalize_citation(item: dict) -> dict:
    source = item.get("source", "") or item.get("url", "") or "unknown"
    return {
        "context_id": item.get("context_id", ""),
        "title": item.get("title", ""),
        "url": item.get("url", ""),
        "source": source,
        "source_type": item.get("source_type", ""),
        "score": float(item.get("score", 0.0) or 0.0),
    }


def _build_context_item(
    item: dict,
    *,
    source_type: str,
    retrieval_stage: str,
    original_rank: int,
    context_id: str,
) -> dict:
    return {
        "context_id": context_id,
        "text": item.get("text", ""),
        "source_type": source_type,
        "retrieval_stage": retrieval_stage,
        "original_rank": original_rank,
        "collection_name": item.get("collection_name", ""),
        "doc_type": item.get(
            "doc_type",
            "web_page" if source_type == "web" else "retrieved_chunk",
        ),
        "chunk_type": item.get(
            "chunk_type",
            "web_snippet" if source_type == "web" else "text_chunk",
        ),
        "modality": item.get("modality", "text"),
        "source_quality": item.get(
            "source_quality",
            "external" if source_type == "web" else "retrieved",
        ),
        "title": item.get("title", ""),
        "url": item.get("url", ""),
        "source": item.get("source", ""),
        "score": float(item.get("score", 0.0) or 0.0),
        "image_path": item.get("image_path", ""),
        "structured_data": item.get("structured_data", {}),
    }


def _citation_sort_key(item: dict) -> tuple[float, int]:
    return (float(item.get("score", 0.0) or 0.0), -int(item.get("original_rank", 0) or 0))


def _extract_citation_ids(answer: str) -> list[str]:
    return _CITATION_PATTERN.findall(answer)


def _render_citation_reference(item: dict) -> str:
    title = item.get("title", "") or "Nguồn tham khảo"
    url = item.get("url", "") or ""
    source = item.get("source", "") or "unknown"
    if url:
        return f"[{title}]({url})"
    return f"{title} ({source})"


def _render_answer_citations(answer: str, citation_pool: dict[str, dict]) -> tuple[str, list[dict]]:
    ordered_citations: list[dict] = []
    seen_context_ids: set[str] = set()

    def replace(match: re.Match[str]) -> str:
        context_id = match.group(1)
        item = citation_pool.get(context_id)
        if item is None:
            return ""
        if context_id not in seen_context_ids:
            seen_context_ids.add(context_id)
            ordered_citations.append(_normalize_citation(item))
        return _render_citation_reference(item)

    rendered = _CITATION_PATTERN.sub(replace, answer)
    rendered = re.sub(r"[ ]{2,}", " ", rendered).strip()
    return rendered, ordered_citations


def _build_citation_repair_prompt(
    answer: str,
    citation_pool: dict[str, dict],
    citation_limit: int,
) -> str:
    top_candidates = sorted(
        citation_pool.values(),
        key=_citation_sort_key,
        reverse=True,
    )[:citation_limit]
    candidate_lines = "\n".join(
        [
            (
                f"- {item['context_id']}: title={item.get('title', '')}; "
                f"source={item.get('source', '')}; url={item.get('url', '')}"
            )
            for item in top_candidates
        ]
    )
    return (
        "Giữ nguyên nội dung câu trả lời sau và chỉ chèn placeholder trích dẫn inline "
        "ngay sau các claim dạng [[cite:context_id]]. Không đổi wording ngoài việc thêm placeholder.\n\n"
        f"Các context_id hợp lệ:\n{candidate_lines}\n\n"
        f"Câu trả lời hiện tại:\n{answer}"
    )


def _build_generation_prompt(state: RAGState, config) -> str:
    retrieved_context = "\n\n".join(
        [
            (
                f"Context ID: {c.get('context_id', '')}\n"
                f"Title: {c.get('title', 'N/A')}\n"
                f"Source: {c.get('source', '')}\n"
                f"URL: {c.get('url', '')}\n"
                f"Content: {c.get('text', '')}"
            )
            for c in state["final_context"][: config.rag.context_limit]
        ]
    )
    context_sections = []
    if state.get("conversation_context"):
        context_sections.append(state["conversation_context"])
    context_sections.append(
        "Quy tắc trích dẫn: khi dùng nguồn, chèn placeholder inline ngay sau claim theo dạng [[cite:context_id]]. "
        "Chỉ dùng context_id có trong ngữ cảnh. Không dùng [1] hoặc danh sách nguồn cuối câu trả lời."
    )
    if retrieved_context:
        context_sections.append(retrieved_context)

    return config.prompts.user_template.format(
        context="\n\n".join(context_sections),
        question=_raw_query(state),
    )


def _build_retry_prompt(state: RAGState, guard_result: dict) -> str:
    categories = ", ".join(guard_result.get("categories", [])) or "Unknown"
    refusal = guard_result.get("refusal") or "Unknown"
    safe_label = guard_result.get("safe_label") or "Unsafe"
    return (
        f"{state['generation_prompt']}\n\n"
        "Ban nhap truoc da bi bo loc an toan danh dau khong an toan.\n"
        f"Safety: {safe_label}\n"
        f"Categories: {categories}\n"
        f"Refusal: {refusal}\n\n"
        f"Ban nhap khong an toan:\n{state['answer']}\n\n"
        "Hay tao lai cau tra loi bang tieng Viet, ngan gon, huu ich, an toan, va khong bao gom noi dung thuoc cac nhom tren. "
        "Neu khong the tra loi an toan, hay tu choi ngan gon."
    )


def build_rag_graph(services, config):
    """Build and compile the LangGraph RAG workflow."""

    @traceable(name="Input Guard Check")
    async def input_guard_node(state: RAGState) -> dict:
        guard_result = await services.guard.check_input(_resolved_query(state))
        if guard_result["label"] != "safe":
            return {
                "input_safe": False,
                "input_guard_result": guard_result,
                "answer": _build_denial_message(
                    config.prompts.apology_message,
                    guard_result.get("categories", []),
                ),
            }
        return {"input_safe": True, "input_guard_result": guard_result}

    @traceable(name="Query Embedding")
    async def embed_node(state: RAGState) -> dict:
        try:
            embeddings = await services.embedder.embed_query(_resolved_query(state))
            return {"embeddings": embeddings}
        except Exception as e:
            logger.error(f"Embedding failed: {e}")
            return {"error": str(e)}

    @traceable(name="Document Retrieval")
    async def retrieve_node(state: RAGState) -> dict:
        sparse_vector = None
        retrieval_mode = "dense_only"
        try:
            sparse_vector = services.sparse_encoder.encode_query(_resolved_query(state))
            retrieval_mode = "hybrid"
        except Exception as e:
            logger.warning(
                f"Sparse encoding failed, falling back to dense-only retrieval: {e}"
            )

        docs = await services.retriever.hybrid_search(
            dense_vector=state["embeddings"],
            sparse_vector=sparse_vector,
            top_k=config.rag.retrieval_top_k,
        )
        logger.log("RETRIEVAL", f"retrieval_mode={retrieval_mode}")
        return {"retrieved_docs": docs}

    @traceable(name="Document Reranking")
    async def rerank_node(state: RAGState) -> dict:
        if not state["retrieved_docs"]:
            return {"reranked_docs": []}
        passages = [d["text"] for d in state["retrieved_docs"]]
        ranked = await services.reranker.rerank(
            query=_resolved_query(state),
            passages=passages,
            top_n=config.rag.rerank_top_n,
            instruction=config.prompts.reranker_instruction,
        )
        return {"reranked_docs": ranked}

    @traceable(name="Web Search Fallback")
    async def web_fallback_node(state: RAGState) -> dict:
        needs_fallback = (
            len(state["reranked_docs"]) < config.rag.fallback_min_chunks
            or (
                state["reranked_docs"]
                and state["reranked_docs"][0]["score"] < config.rag.fallback_score_threshold
            )
        )
        if needs_fallback:
            web_results = await services.web_search.search(_resolved_query(state))
            return {"web_results": web_results}
        return {"web_results": []}

    @traceable(name="Combine Context")
    async def combine_context_node(state: RAGState) -> dict:
        reranked = []
        for rank, ranked_doc in enumerate(state["reranked_docs"]):
            index = ranked_doc.get("index", -1)
            if 0 <= index < len(state["retrieved_docs"]):
                merged = dict(state["retrieved_docs"][index])
                merged["score"] = ranked_doc.get("score", merged.get("score", 0.0))
                reranked.append(
                    _build_context_item(
                        merged,
                        source_type="hybrid",
                        retrieval_stage="rerank",
                        original_rank=rank,
                        context_id=f"hybrid:{merged.get('id', rank)}",
                    )
                )
        web_context = []
        for rank, web_item in enumerate(state["web_results"]):
            web_context.append(
                _build_context_item(
                    web_item,
                    source_type="web",
                    retrieval_stage=web_item.get("retrieval_stage", "web_fallback"),
                    original_rank=int(web_item.get("original_rank", rank)),
                    context_id=web_item.get("context_id", f"web:{rank}"),
                )
            )
        final_context = reranked + web_context
        citation_pool = {
            item["context_id"]: item
            for item in final_context
            if item.get("context_id")
        }
        return {"final_context": final_context, "citation_pool": citation_pool}

    @traceable(name="LLM Generation")
    async def generate_node(state: RAGState) -> dict:
        if not state["final_context"]:
            return {"answer": config.prompts.no_context_message, "generation_prompt": ""}

        user_prompt = _build_generation_prompt(state, config)
        answer = await services.llm.generate(
            system_prompt=config.prompts.system_prompt,
            user_prompt=user_prompt,
        )
        return {"answer": answer, "generation_prompt": user_prompt}

    @traceable(name="Output Safety Check")
    async def output_guard_node(state: RAGState) -> dict:
        guard_result = await services.guard.check_output(
            text=state["answer"],
            prompt=_resolved_query(state),
        )
        if guard_result["label"] == "safe":
            return {"output_safe": True, "output_guard_result": guard_result}

        if state.get("generation_prompt"):
            retry_prompt = _build_retry_prompt(state, guard_result)
            retry_answer = await services.llm.generate(
                system_prompt=config.prompts.system_prompt,
                user_prompt=retry_prompt,
            )
            retry_guard_result = await services.guard.check_output(
                text=retry_answer,
                prompt=_resolved_query(state),
            )
            if retry_guard_result["label"] == "safe":
                return {
                    "answer": retry_answer,
                    "output_safe": True,
                    "output_guard_result": retry_guard_result,
                }
            guard_result = retry_guard_result

        return {
            "output_safe": False,
            "output_guard_result": guard_result,
            "answer": _build_denial_message(
                config.prompts.guard_error_message,
                guard_result.get("categories", []),
            ),
        }

    @traceable(name="Build Citations")
    async def citations_node(state: RAGState) -> dict:
        citation_pool = state.get("citation_pool", {})
        candidates = sorted(
            citation_pool.values(),
            key=_citation_sort_key,
            reverse=True,
        )
        valid_citation_ids = [
            context_id
            for context_id in _extract_citation_ids(state["answer"])
            if context_id in citation_pool
        ]
        answer = state["answer"]

        if not valid_citation_ids and citation_pool:
            repair_prompt = _build_citation_repair_prompt(
                state["answer"],
                citation_pool,
                config.rag.citation_limit,
            )
            try:
                repaired_answer = await services.llm.generate(
                    system_prompt=config.prompts.system_prompt,
                    user_prompt=repair_prompt,
                )
            except Exception as e:
                logger.warning("Citation repair failed before placeholder rendering: {}", e)
            else:
                repaired_ids = [
                    context_id
                    for context_id in _extract_citation_ids(repaired_answer)
                    if context_id in citation_pool
                ]
                if repaired_ids:
                    answer = repaired_answer
                    valid_citation_ids = repaired_ids

        if not valid_citation_ids:
            fallback_items = candidates[: config.rag.citation_limit]
            citations = [_normalize_citation(item) for item in fallback_items]
            if fallback_items:
                logger.warning("Citation repair failed; using deterministic fallback citations")
                fallback_refs = " ".join(
                    _render_citation_reference(item) for item in fallback_items
                )
                answer = f"{state['answer'].rstrip()} {fallback_refs}".strip()
            return {"answer": answer, "citations": citations}

        answer, citations = _render_answer_citations(answer, citation_pool)
        return {"answer": answer, "citations": citations}

    def route_after_input_guard(state: RAGState) -> Literal["embed", "__end__"]:
        return "embed" if state["input_safe"] else "__end__"

    def route_after_embed(state: RAGState) -> Literal["retrieve", "__end__"]:
        return "__end__" if state.get("error") else "retrieve"

    def route_after_output_guard(state: RAGState) -> Literal["citations", "__end__"]:
        return "citations" if state["output_safe"] else "__end__"

    workflow = StateGraph(RAGState)

    workflow.add_node("input_guard", input_guard_node)
    workflow.add_node("embed", embed_node)
    workflow.add_node("retrieve", retrieve_node)
    workflow.add_node("rerank", rerank_node)
    workflow.add_node("web_fallback", web_fallback_node)
    workflow.add_node("combine_context", combine_context_node)
    workflow.add_node("generate", generate_node)
    workflow.add_node("output_guard", output_guard_node)
    workflow.add_node("citations", citations_node)

    workflow.add_edge(START, "input_guard")
    workflow.add_conditional_edges("input_guard", route_after_input_guard)
    workflow.add_conditional_edges("embed", route_after_embed)
    workflow.add_edge("retrieve", "rerank")
    workflow.add_edge("rerank", "web_fallback")
    workflow.add_edge("web_fallback", "combine_context")
    workflow.add_edge("combine_context", "generate")
    workflow.add_edge("generate", "output_guard")
    workflow.add_conditional_edges("output_guard", route_after_output_guard)
    workflow.add_edge("citations", END)

    return workflow.compile()
