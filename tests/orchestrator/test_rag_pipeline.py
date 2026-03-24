import pytest
from unittest.mock import AsyncMock, MagicMock


SAFE_RESULT = {"label": "safe", "safe_label": "Safe", "categories": [], "refusal": None}
SAFE_OUTPUT_RESULT = {"label": "safe", "safe_label": "Safe", "categories": [], "refusal": "No"}
UNSAFE_VIOLENT_RESULT = {
    "label": "unsafe",
    "safe_label": "Unsafe",
    "categories": ["Violent"],
    "refusal": "Yes",
}


def _make_services(
    input_guard_result=None,
    embed_vector=None,
    embed_raises=False,
    sparse_vector=None,
    sparse_encode_raises=False,
    retrieved_docs=None,
    reranked=None,
    web_results=None,
    llm_side_effect=None,
    output_guard_result=None,
    output_guard_side_effect=None,
):
    """Build a mock services object for RAG pipeline tests."""
    services = MagicMock()
    services.guard.check_input = AsyncMock(return_value=input_guard_result or SAFE_RESULT)
    if output_guard_side_effect is not None:
        services.guard.check_output = AsyncMock(side_effect=output_guard_side_effect)
    else:
        services.guard.check_output = AsyncMock(return_value=output_guard_result or SAFE_OUTPUT_RESULT)
    if embed_raises:
        services.embedder.embed_query = AsyncMock(side_effect=Exception("embed down"))
    else:
        services.embedder.embed_query = AsyncMock(return_value=embed_vector or [0.1] * 1024)
    if sparse_encode_raises:
        services.sparse_encoder.encode_query = MagicMock(side_effect=Exception("sparse down"))
    else:
        services.sparse_encoder.encode_query = MagicMock(
            return_value=sparse_vector or {"indices": [1, 2], "values": [0.3, 0.7]}
        )
    services.retriever.hybrid_search = AsyncMock(return_value=retrieved_docs or [])
    services.reranker.rerank = AsyncMock(return_value=reranked or [])
    services.web_search.search = AsyncMock(return_value=web_results or [])
    services.llm.generate = AsyncMock(side_effect=llm_side_effect or ["Test answer"])
    return services


def _make_config(
    retrieval_top_k=20,
    rerank_top_n=5,
    fallback_min_chunks=3,
    fallback_score_threshold=0.5,
    no_context_message="Xin lỗi, tôi không tìm thấy thông tin liên quan.",
    guard_error_message="Xin lỗi, yêu cầu của bạn không thể xử lý.",
    apology_message="Xin lỗi, tôi không thể trả lời câu hỏi này theo nội dung của chúng tôi.",
):
    config = MagicMock()
    config.rag = MagicMock()
    config.rag.retrieval_top_k = retrieval_top_k
    config.rag.rerank_top_n = rerank_top_n
    config.rag.fallback_min_chunks = fallback_min_chunks
    config.rag.fallback_score_threshold = fallback_score_threshold
    config.rag.context_limit = 5
    config.rag.citation_limit = 5
    config.prompts = MagicMock()
    config.prompts.system_prompt = "You are a test assistant."
    config.prompts.user_template = "Context:\n{context}\n\nQuestion: {question}"
    config.prompts.no_context_message = no_context_message
    config.prompts.guard_error_message = guard_error_message
    config.prompts.apology_message = apology_message
    config.prompts.reranker_instruction = "Rank by relevance."
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
        "input_guard_result": {},
        "output_guard_result": {},
        "generation_prompt": "",
        "citations": [],
        "error": None,
    }


@pytest.mark.asyncio
async def test_rag_pipeline_happy_path():
    """Full happy path: all services succeed, answer + citations returned."""
    from orchestrator.pipeline.rag import build_rag_graph

    retrieved = [{
        "id": "1",
        "text": "GDP text",
        "source": "src.com",
        "title": "GDP",
        "url": "https://example.com/gdp",
        "score": 0.7,
    }]
    reranked = [{"index": 0, "score": 0.9}]

    services = _make_services(
        retrieved_docs=retrieved,
        reranked=reranked,
        llm_side_effect=["GDP tăng 7%"],
    )
    config = _make_config(fallback_min_chunks=1)
    graph = build_rag_graph(services, config)

    result = await graph.ainvoke(_initial_state())

    assert result["input_safe"] is True
    assert result["answer"] == "GDP tăng 7%"
    assert result["output_safe"] is True
    assert len(result["citations"]) == 1
    assert result["citations"][0]["title"] == "GDP"
    assert result["citations"][0]["url"] == "https://example.com/gdp"
    assert result["citations"][0]["score"] == 0.9
    services.llm.generate.assert_called_once()
    services.web_search.search.assert_not_called()


@pytest.mark.asyncio
async def test_rag_pipeline_unsafe_input_uses_category_reason():
    """Unsafe input returns informative denial text and skips generation."""
    from orchestrator.pipeline.rag import build_rag_graph

    services = _make_services(input_guard_result=UNSAFE_VIOLENT_RESULT)
    config = _make_config()
    graph = build_rag_graph(services, config)

    result = await graph.ainvoke(_initial_state())

    assert result["input_safe"] is False
    assert result["answer"].startswith(config.prompts.apology_message)
    assert "bạo lực" in result["answer"].lower()
    services.llm.generate.assert_not_called()
    services.embedder.embed_query.assert_not_called()


@pytest.mark.asyncio
async def test_rag_pipeline_no_context():
    """No context: empty retrieval + no web results -> no_context_message, LLM not called."""
    from orchestrator.pipeline.rag import build_rag_graph

    services = _make_services(
        retrieved_docs=[],
        reranked=[],
        web_results=[],
    )
    config = _make_config(fallback_min_chunks=0)
    graph = build_rag_graph(services, config)

    result = await graph.ainvoke(_initial_state())

    assert result["answer"] == config.prompts.no_context_message
    services.llm.generate.assert_not_called()


@pytest.mark.asyncio
async def test_rag_pipeline_web_fallback_triggered():
    """Web fallback: 1 reranked doc < fallback_min_chunks=3 -> web search called."""
    from orchestrator.pipeline.rag import build_rag_graph

    retrieved = [{
        "id": "1",
        "text": "t",
        "source": "s",
        "title": "T",
        "url": "https://example.com/t",
        "score": 0.9,
    }]
    reranked = [{"index": 0, "score": 0.9}]
    web = [{
        "text": "web content",
        "source": "web.com",
        "title": "Web",
        "url": "https://web.com/article",
        "score": 0.42,
    }]

    services = _make_services(
        retrieved_docs=retrieved,
        reranked=reranked,
        web_results=web,
        llm_side_effect=["Answer with web"],
    )
    config = _make_config(fallback_min_chunks=3)
    graph = build_rag_graph(services, config)

    result = await graph.ainvoke(_initial_state())

    services.web_search.search.assert_called_once()
    assert any(c["source"] == "web.com" for c in result["final_context"])
    assert result["answer"] == "Answer with web"


@pytest.mark.asyncio
async def test_rag_pipeline_embed_failure():
    """Embed failure: error set -> retrieve and web search never called."""
    from orchestrator.pipeline.rag import build_rag_graph

    services = _make_services(embed_raises=True)
    config = _make_config()
    graph = build_rag_graph(services, config)

    result = await graph.ainvoke(_initial_state())

    assert result["error"] is not None
    assert result["answer"] == ""
    services.retriever.hybrid_search.assert_not_called()
    services.web_search.search.assert_not_called()
    services.llm.generate.assert_not_called()


@pytest.mark.asyncio
async def test_rag_pipeline_passes_sparse_vector_to_hybrid_search():
    """Retrieval uses sparse query encoding and passes it to hybrid search."""
    from orchestrator.pipeline.rag import build_rag_graph

    services = _make_services(
        sparse_vector={"indices": [3, 9], "values": [0.2, 0.8]},
        retrieved_docs=[],
        reranked=[],
        web_results=[],
    )
    config = _make_config(fallback_min_chunks=0)
    graph = build_rag_graph(services, config)

    await graph.ainvoke(_initial_state(query="lạm phát là gì?"))

    services.sparse_encoder.encode_query.assert_called_once_with("lạm phát là gì?")
    services.retriever.hybrid_search.assert_awaited_once_with(
        dense_vector=[0.1] * 1024,
        sparse_vector={"indices": [3, 9], "values": [0.2, 0.8]},
        top_k=config.rag.retrieval_top_k,
    )


@pytest.mark.asyncio
async def test_rag_pipeline_falls_back_to_dense_only_when_sparse_encoding_fails():
    """Sparse encoder failure should not break retrieval; dense-only fallback is used."""
    from orchestrator.pipeline.rag import build_rag_graph

    services = _make_services(
        sparse_encode_raises=True,
        retrieved_docs=[],
        reranked=[],
        web_results=[],
    )
    config = _make_config(fallback_min_chunks=0)
    graph = build_rag_graph(services, config)

    await graph.ainvoke(_initial_state(query="CPI hôm nay"))

    services.sparse_encoder.encode_query.assert_called_once_with("CPI hôm nay")
    services.retriever.hybrid_search.assert_awaited_once_with(
        dense_vector=[0.1] * 1024,
        sparse_vector=None,
        top_k=config.rag.retrieval_top_k,
    )


@pytest.mark.asyncio
async def test_rag_pipeline_unsafe_output_regenerates_once():
    """Unsafe output triggers one regeneration guided by categories."""
    from orchestrator.pipeline.rag import build_rag_graph

    retrieved = [{
        "id": "1",
        "text": "t",
        "source": "s",
        "title": "T",
        "url": "https://example.com/t",
        "score": 0.9,
    }]
    reranked = [{"index": 0, "score": 0.9}]

    services = _make_services(
        retrieved_docs=retrieved,
        reranked=reranked,
        llm_side_effect=["Unsafe generated text", "Safe regenerated text"],
        output_guard_side_effect=[UNSAFE_VIOLENT_RESULT, SAFE_OUTPUT_RESULT],
    )
    config = _make_config(fallback_min_chunks=1)
    graph = build_rag_graph(services, config)

    result = await graph.ainvoke(_initial_state())

    assert result["output_safe"] is True
    assert result["answer"] == "Safe regenerated text"
    assert services.llm.generate.await_count == 2
    assert len(result["citations"]) == 1
    retry_prompt = services.llm.generate.await_args_list[1].kwargs["user_prompt"]
    assert "Violent" in retry_prompt
    assert "Unsafe generated text" in retry_prompt


@pytest.mark.asyncio
async def test_rag_pipeline_denies_after_unsafe_retry():
    """A second unsafe output returns an informative denial and no citations."""
    from orchestrator.pipeline.rag import build_rag_graph

    retrieved = [{
        "id": "1",
        "text": "t",
        "source": "s",
        "title": "T",
        "url": "https://example.com/t",
        "score": 0.9,
    }]
    reranked = [{"index": 0, "score": 0.9}]

    services = _make_services(
        retrieved_docs=retrieved,
        reranked=reranked,
        llm_side_effect=["Unsafe generated text", "Still unsafe text"],
        output_guard_side_effect=[UNSAFE_VIOLENT_RESULT, UNSAFE_VIOLENT_RESULT],
    )
    config = _make_config(fallback_min_chunks=1)
    graph = build_rag_graph(services, config)

    result = await graph.ainvoke(_initial_state())

    assert result["output_safe"] is False
    assert result["answer"].startswith(config.prompts.guard_error_message)
    assert "bạo lực" in result["answer"].lower()
    assert result["citations"] == []
