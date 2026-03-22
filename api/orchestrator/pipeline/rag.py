from __future__ import annotations

from langgraph.graph import StateGraph, START, END
from typing import TypedDict, Literal
from langsmith import traceable
from loguru import logger


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
    citations: list[dict]
    error: str | None


def build_rag_graph(services, config):
    """Build and compile the LangGraph RAG workflow.

    Args:
        services: Object with attributes: guard, embedder, retriever, reranker, llm, web_search
        config: Settings object with RAG params and messages

    Returns:
        Compiled LangGraph app (call with ainvoke(initial_state))
    """

    @traceable(name="Input Guard Check")
    async def input_guard_node(state: RAGState) -> dict:
        safe = await services.guard.check_input(state["query"])
        if not safe:
            return {
                "input_safe": False,
                "answer": config.prompts.apology_message,
            }
        return {"input_safe": True}

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
        reranked = [
            state["retrieved_docs"][r["index"]]
            for r in state["reranked_docs"]
            if 0 <= r["index"] < len(state["retrieved_docs"])
        ]
        final_context = reranked + state["web_results"]
        return {"final_context": final_context}

    @traceable(name="LLM Generation")
    async def generate_node(state: RAGState) -> dict:
        if not state["final_context"]:
            return {"answer": config.prompts.no_context_message}

        context_text = "\n\n".join([
            f"[{c.get('title', 'N/A')}] {c.get('text', '')}"
            for c in state["final_context"][:config.rag.context_limit]
        ])
        user_prompt = config.prompts.user_template.format(
            context=context_text,
            question=state["query"],
        )
        answer = await services.llm.generate(
            system_prompt=config.prompts.system_prompt,
            user_prompt=user_prompt,
        )
        return {"answer": answer}

    @traceable(name="Output Safety Check")
    async def output_guard_node(state: RAGState) -> dict:
        output_safe = await services.guard.check_output(
            text=state["answer"],
            prompt=state["query"],
        )
        if not output_safe:
            return {
                "output_safe": False,
                "answer": config.prompts.guard_error_message,
            }
        return {"output_safe": True}

    @traceable(name="Build Citations")
    async def citations_node(state: RAGState) -> dict:
        citations = [
            {"title": c.get("title", ""), "source": c.get("source", "")}
            for c in state["final_context"][:config.rag.citation_limit]
        ]
        return {"citations": citations}

    # --- Routing functions ---

    def route_after_input_guard(state: RAGState) -> Literal["embed", "__end__"]:
        return "embed" if state["input_safe"] else "__end__"

    def route_after_embed(state: RAGState) -> Literal["retrieve", "__end__"]:
        return "__end__" if state.get("error") else "retrieve"

    def route_after_output_guard(state: RAGState) -> Literal["citations", "__end__"]:
        return "citations" if state["output_safe"] else "__end__"

    # --- Build graph ---
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
