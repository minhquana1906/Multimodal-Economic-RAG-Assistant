from __future__ import annotations

from typing import Literal, TypedDict

from langgraph.graph import END, START, StateGraph
from loguru import logger

from orchestrator.pipeline.rag_context import (
    combine_context_sources,
    finalize_citations,
)
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
    task_type: str
    embeddings: list[float]
    retrieved_docs: list[dict]
    reranked_docs: list[dict]
    web_results: list[dict]
    final_context: list[dict]
    answer: str
    generation_prompt: str
    citations: list[dict]
    citation_pool: dict[str, dict]
    error: str | None


def _resolved_query(state: RAGState) -> str:
    return state.get("resolved_query") or state.get("raw_query") or state.get("query", "")


def build_rag_graph(services, config, *, retrieval_only: bool = False):
    """Build and compile the LangGraph RAG workflow.

    retrieval_only=True stops after combine_context — use for streaming
    where generation is streamed outside the graph.
    """

    async def embed_node(state: RAGState) -> dict:
        try:
            embeddings = await services.inference.embed_query(_resolved_query(state))
            return {"embeddings": embeddings}
        except Exception as e:
            logger.error(f"Embedding failed: {e}")
            return {"error": str(e)}

    async def retrieve_node(state: RAGState) -> dict:
        sparse_vector = None
        retrieval_mode = "dense_only"
        try:
            sparse_vector = await services.inference.sparse_query(_resolved_query(state))
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
        scores = await services.inference.rerank(
            query=_resolved_query(state),
            passages=passages,
        )
        ranked = sorted(
            [{"index": i, "score": s} for i, s in enumerate(scores)],
            key=lambda x: x["score"],
            reverse=True,
        )
        return {"reranked_docs": ranked[: config.rag.rerank_top_n]}

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

    async def citations_node(state: RAGState) -> dict:
        return finalize_citations(
            state,
            context_limit=config.rag.context_limit,
            citation_limit=config.rag.citation_limit,
        )

    def route_after_embed(state: RAGState) -> Literal["retrieve", "__end__"]:
        return "__end__" if state.get("error") else "retrieve"

    workflow = StateGraph(RAGState)

    workflow.add_node("embed", embed_node)
    workflow.add_node("retrieve", retrieve_node)
    workflow.add_node("rerank", rerank_node)
    workflow.add_node("web_fallback", web_fallback_node)
    workflow.add_node("combine_context", combine_context_node)

    workflow.add_edge(START, "embed")
    workflow.add_conditional_edges("embed", route_after_embed)
    workflow.add_edge("retrieve", "rerank")
    workflow.add_edge("rerank", "web_fallback")
    workflow.add_edge("web_fallback", "combine_context")

    if retrieval_only:
        workflow.add_edge("combine_context", END)
    else:
        workflow.add_node("generate", generate_node)
        workflow.add_node("citations", citations_node)
        workflow.add_edge("combine_context", "generate")
        workflow.add_edge("generate", "citations")
        workflow.add_edge("citations", END)

    return workflow.compile()
