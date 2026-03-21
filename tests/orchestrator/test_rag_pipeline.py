import pytest
from unittest.mock import AsyncMock, MagicMock


def _make_services(
    input_safe=True,
    embed_vector=None,
    embed_raises=False,
    retrieved_docs=None,
    reranked=None,
    web_results=None,
    answer="Test answer",
    output_safe=True,
):
    """Build a mock services object for RAG pipeline tests."""
    services = MagicMock()

    services.guard.check_input = AsyncMock(return_value=input_safe)
    services.guard.check_output = AsyncMock(return_value=output_safe)

    if embed_raises:
        services.embedder.embed_query = AsyncMock(side_effect=Exception("embed down"))
    else:
        services.embedder.embed_query = AsyncMock(return_value=embed_vector or [0.1] * 1024)

    services.retriever.hybrid_search = AsyncMock(return_value=retrieved_docs or [])
    services.reranker.rerank = AsyncMock(return_value=reranked or [])
    services.web_search.search = AsyncMock(return_value=web_results or [])
    services.llm.generate = AsyncMock(return_value=answer)

    return services


def _make_config(
    retrieval_top_k=20,
    rerank_top_n=5,
    fallback_min_chunks=3,
    fallback_score_threshold=0.5,
    no_context_message="Xin lỗi, tôi không tìm thấy thông tin liên quan.",
    guard_error_message="Xin lỗi, yêu cầu của bạn không thể xử lý.",
):
    config = MagicMock()
    config.retrieval_top_k = retrieval_top_k
    config.rerank_top_n = rerank_top_n
    config.fallback_min_chunks = fallback_min_chunks
    config.fallback_score_threshold = fallback_score_threshold
    config.no_context_message = no_context_message
    config.guard_error_message = guard_error_message
    return config


def _initial_state(query="GDP Việt Nam?"):
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
        "citations": [],
        "error": None,
    }


@pytest.mark.asyncio
async def test_rag_pipeline_happy_path():
    """Happy path: input safe → embed → retrieve → rerank → generate → output safe → citations."""
    from orchestrator.pipeline.rag import build_rag_graph

    retrieved = [{"id": "1", "text": "GDP text", "source": "src.com", "title": "GDP", "score": 0.9}]
    reranked = [{"index": 0, "score": 0.9}]

    services = _make_services(
        input_safe=True,
        retrieved_docs=retrieved,
        reranked=reranked,
        answer="GDP tăng 7%",
        output_safe=True,
    )
    config = _make_config()
    graph = await build_rag_graph(services, config)

    result = await graph.ainvoke(_initial_state())

    assert result["input_safe"] is True
    assert result["answer"] == "GDP tăng 7%"
    assert result["output_safe"] is True
    assert len(result["citations"]) == 1
    assert result["citations"][0]["title"] == "GDP"
    services.llm.generate.assert_called_once()


@pytest.mark.asyncio
async def test_rag_pipeline_unsafe_input():
    """Unsafe input: guard returns False → pipeline ends early, LLM never called."""
    from orchestrator.pipeline.rag import build_rag_graph

    services = _make_services(input_safe=False)
    config = _make_config()
    graph = await build_rag_graph(services, config)

    result = await graph.ainvoke(_initial_state())

    assert result["input_safe"] is False
    assert result["answer"] == ""          # LLM never called
    services.llm.generate.assert_not_called()
    services.embedder.embed_query.assert_not_called()


@pytest.mark.asyncio
async def test_rag_pipeline_no_context():
    """No context found: final_context empty → no_context_message returned."""
    from orchestrator.pipeline.rag import build_rag_graph

    services = _make_services(
        input_safe=True,
        retrieved_docs=[],
        reranked=[],
        web_results=[],
    )
    config = _make_config()
    graph = await build_rag_graph(services, config)

    result = await graph.ainvoke(_initial_state())

    # generate_node returns no_context_message when final_context is empty
    assert result["answer"] == config.no_context_message
    # LLM generate should not be called (no_context_message is set directly)
    services.llm.generate.assert_not_called()


@pytest.mark.asyncio
async def test_rag_pipeline_web_fallback_triggered():
    """Web fallback: reranked_docs < fallback_min_chunks → web search called."""
    from orchestrator.pipeline.rag import build_rag_graph

    retrieved = [{"id": "1", "text": "t", "source": "s", "title": "T", "score": 0.9}]
    reranked = [{"index": 0, "score": 0.9}]   # only 1, threshold is 3
    web = [{"text": "web content", "source": "web.com", "title": "Web"}]

    services = _make_services(
        input_safe=True,
        retrieved_docs=retrieved,
        reranked=reranked,       # 1 < fallback_min_chunks=3 → triggers fallback
        web_results=web,
        answer="Answer with web",
        output_safe=True,
    )
    config = _make_config(fallback_min_chunks=3)
    graph = await build_rag_graph(services, config)

    result = await graph.ainvoke(_initial_state())

    services.web_search.search.assert_called_once()
    # final_context should include both reranked + web results
    assert any(c["source"] == "web.com" for c in result["final_context"])
    assert result["answer"] == "Answer with web"
