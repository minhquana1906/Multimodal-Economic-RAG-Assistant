from __future__ import annotations

from typing import Literal, TypedDict

from langgraph.graph import END, START, StateGraph
from loguru import logger

from orchestrator.pipeline.rag_context import (
    combine_context_sources,
    finalize_citations,
)
from orchestrator.pipeline.rag_guard import build_denial_message, build_retry_prompt
from orchestrator.pipeline.rag_policy import should_add_web_fallback
from orchestrator.pipeline.rag_prompts import (
    build_generation_prompt,
    resolve_rag_system_prompt,
)


class RAGState(TypedDict):
    """Complete RAG pipeline state passed between nodes."""

    query: str
    raw_query: str
    resolved_query: str
    conversation_summary: str
    conversation_context: str
    task_type: str
    response_mode: Literal["text", "audio"]
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


def _resolved_query(state: RAGState) -> str:
    return state.get("resolved_query") or state.get("raw_query") or state.get("query", "")


def build_rag_graph(services, config):
    """Build and compile the LangGraph RAG workflow."""

    async def input_guard_node(state: RAGState) -> dict:
        guard_result = await services.guard.check_input(_resolved_query(state))
        if guard_result["label"] != "safe":
            return {
                "input_safe": False,
                "input_guard_result": guard_result,
                "answer": build_denial_message(
                    config.prompts.apology_message,
                    guard_result.get("categories", []),
                ),
            }
        return {"input_safe": True, "input_guard_result": guard_result}

    async def embed_node(state: RAGState) -> dict:
        try:
            embeddings = await services.embedder.embed_query(_resolved_query(state))
            return {"embeddings": embeddings}
        except Exception as e:
            logger.error(f"Embedding failed: {e}")
            return {"error": str(e)}

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

    async def web_fallback_node(state: RAGState) -> dict:
        if should_add_web_fallback(state, config):
            web_results = await services.web_search.search(_resolved_query(state))
            return {"web_results": web_results}
        return {"web_results": []}

    async def combine_context_node(state: RAGState) -> dict:
        return combine_context_sources(
            retrieved_docs=state["retrieved_docs"],
            reranked_docs=state["reranked_docs"],
            web_results=state["web_results"],
        )

    async def generate_node(state: RAGState) -> dict:
        if not state["final_context"]:
            return {
                "answer": config.prompts.no_context_message,
                "generation_prompt": "",
            }

        user_prompt = build_generation_prompt(state, config)
        system_prompt = resolve_rag_system_prompt(config.prompts)
        answer = await services.llm.generate(
            system_prompt=system_prompt,
            user_prompt=user_prompt,
        )
        return {"answer": answer, "generation_prompt": user_prompt}

    async def output_guard_node(state: RAGState) -> dict:
        guard_result = await services.guard.check_output(
            text=state["answer"],
            prompt=_resolved_query(state),
        )
        if guard_result["label"] == "safe":
            return {"output_safe": True, "output_guard_result": guard_result}

        if state.get("generation_prompt"):
            retry_prompt = build_retry_prompt(
                original_prompt=state["generation_prompt"],
                unsafe_answer=state["answer"],
                guard_result=guard_result,
            )
            system_prompt = resolve_rag_system_prompt(config.prompts)
            retry_answer = await services.llm.generate(
                system_prompt=system_prompt,
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
            "answer": build_denial_message(
                config.prompts.guard_error_message,
                guard_result.get("categories", []),
            ),
        }

    async def citations_node(state: RAGState) -> dict:
        return finalize_citations(
            state,
            citation_limit=config.rag.citation_limit,
        )

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
