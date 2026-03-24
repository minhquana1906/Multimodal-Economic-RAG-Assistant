from __future__ import annotations

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


class RAGState(TypedDict):
    """Complete RAG pipeline state passed between nodes."""

    query: str
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
    error: str | None


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
        "title": item.get("title", ""),
        "url": item.get("url", ""),
        "source": source,
        "score": float(item.get("score", 0.0) or 0.0),
    }


def _build_generation_prompt(state: RAGState, config) -> str:
    context_text = "\n\n".join(
        [
            f"[{c.get('title', 'N/A')}] {c.get('text', '')}"
            for c in state["final_context"][: config.rag.context_limit]
        ]
    )
    return config.prompts.user_template.format(
        context=context_text,
        question=state["query"],
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
        guard_result = await services.guard.check_input(state["query"])
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
            embeddings = await services.embedder.embed_query(state["query"])
            return {"embeddings": embeddings}
        except Exception as e:
            logger.error("Embedding failed: {}", e)
            return {"error": str(e)}

    @traceable(name="Document Retrieval")
    async def retrieve_node(state: RAGState) -> dict:
        docs = await services.retriever.hybrid_search(
            dense_vector=state["embeddings"],
            top_k=config.rag.retrieval_top_k,
        )
        return {"retrieved_docs": docs}

    @traceable(name="Document Reranking")
    async def rerank_node(state: RAGState) -> dict:
        if not state["retrieved_docs"]:
            return {"reranked_docs": []}
        passages = [d["text"] for d in state["retrieved_docs"]]
        ranked = await services.reranker.rerank(
            query=state["query"],
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
            web_results = await services.web_search.search(state["query"])
            return {"web_results": web_results}
        return {"web_results": []}

    @traceable(name="Combine Context")
    async def combine_context_node(state: RAGState) -> dict:
        reranked = []
        for ranked_doc in state["reranked_docs"]:
            index = ranked_doc.get("index", -1)
            if 0 <= index < len(state["retrieved_docs"]):
                merged = dict(state["retrieved_docs"][index])
                merged["score"] = ranked_doc.get("score", merged.get("score", 0.0))
                reranked.append(merged)
        final_context = reranked + state["web_results"]
        return {"final_context": final_context}

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
            prompt=state["query"],
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
                prompt=state["query"],
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
        citations = [
            _normalize_citation(c)
            for c in state["final_context"][: config.rag.citation_limit]
        ]
        return {"citations": citations}

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
