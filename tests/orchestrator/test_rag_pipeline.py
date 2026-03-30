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
    sparse_raises=False,
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
        services.sparse_encoder.encode_query = MagicMock(
            side_effect=RuntimeError("sparse down")
        )
    else:
        services.sparse_encoder.encode_query = MagicMock(
            return_value=sparse_vector or {"indices": [1, 2], "values": [0.3, 0.7]}
        )
    if sparse_raises:
        services.sparse_encoder.encode_query = MagicMock(
            side_effect=RuntimeError("sparse down")
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
    web_fallback_min_chunks=None,
    web_fallback_hard_threshold=0.7,
    web_fallback_soft_threshold=0.85,
    no_context_message="Không tìm thấy dữ liệu phù hợp trong tài liệu nội bộ hoặc nguồn web hiện có.",
    guard_error_message="Xin lỗi, yêu cầu của bạn không thể xử lý.",
    apology_message="Xin lỗi, tôi không thể trả lời câu hỏi này theo nội dung của chúng tôi.",
):
    config = MagicMock()
    config.rag = MagicMock()
    config.rag.retrieval_top_k = retrieval_top_k
    config.rag.rerank_top_n = rerank_top_n
    config.rag.fallback_min_chunks = fallback_min_chunks
    config.rag.fallback_score_threshold = fallback_score_threshold
    config.rag.web_fallback_min_chunks = (
        fallback_min_chunks
        if web_fallback_min_chunks is None
        else web_fallback_min_chunks
    )
    config.rag.web_fallback_hard_threshold = web_fallback_hard_threshold
    config.rag.web_fallback_soft_threshold = web_fallback_soft_threshold
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
        "raw_query": query,
        "resolved_query": query,
        "conversation_summary": "",
        "conversation_context": "",
        "task_type": "chat",
        "response_mode": "text",
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


def _answer_with_footer(
    body: str,
    *,
    title: str,
    url: str,
    source: str,
    score: float,
) -> str:
    return (
        f"{body}\n\n"
        "----\n\n"
        "### Nguồn trích dẫn\n"
        f"- [{title}]({url}) - **{source} ({score:.4f})**"
    )


@pytest.mark.asyncio
async def test_rag_pipeline_passes_sparse_vector_into_hybrid_search():
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
    sparse_vector = {"indices": [3, 8], "values": [0.33, 0.67]}

    services = _make_services(
        sparse_vector=sparse_vector,
        retrieved_docs=retrieved,
        reranked=reranked,
        llm_side_effect=["GDP tăng 7% [[cite:hybrid:1]]"],
    )
    config = _make_config(fallback_min_chunks=1)
    graph = build_rag_graph(services, config)

    result = await graph.ainvoke(_initial_state())

    assert result["answer"] == _answer_with_footer(
        "GDP tăng 7%",
        title="GDP",
        url="https://example.com/gdp",
        source="src.com",
        score=0.9,
    )
    services.sparse_encoder.encode_query.assert_called_once_with("GDP Việt Nam?")
    assert (
        services.retriever.hybrid_search.await_args.kwargs["sparse_vector"]
        == sparse_vector
    )


@pytest.mark.asyncio
async def test_rag_pipeline_falls_back_to_dense_only_when_sparse_encoder_errors():
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
        sparse_raises=True,
        retrieved_docs=retrieved,
        reranked=reranked,
        llm_side_effect=["GDP tăng 7% [[cite:hybrid:1]]"],
    )
    config = _make_config(fallback_min_chunks=1)
    graph = build_rag_graph(services, config)

    result = await graph.ainvoke(_initial_state())

    assert result["answer"] == _answer_with_footer(
        "GDP tăng 7%",
        title="GDP",
        url="https://example.com/gdp",
        source="src.com",
        score=0.9,
    )
    services.sparse_encoder.encode_query.assert_called_once_with("GDP Việt Nam?")
    assert services.retriever.hybrid_search.await_args.kwargs["sparse_vector"] is None


@pytest.mark.asyncio
async def test_rag_pipeline_uses_resolved_query_for_retrieval_and_guards():
    from orchestrator.pipeline.rag import build_rag_graph

    retrieved = [{
        "id": "1",
        "text": "bond text",
        "source": "src.com",
        "title": "Bonds",
        "url": "https://example.com/bonds",
        "score": 0.7,
    }]
    reranked = [{"index": 0, "score": 0.9}]
    raw_query = "Còn trái phiếu doanh nghiệp thì sao?"
    resolved_query = (
        "Trái phiếu doanh nghiệp tại Việt Nam đang chịu tác động thế nào trong bối cảnh thị trường vốn chịu áp lực?"
    )

    services = _make_services(
        retrieved_docs=retrieved,
        reranked=reranked,
        llm_side_effect=["GDP tăng 7% [[cite:hybrid:1]]"],
    )
    config = _make_config(fallback_min_chunks=1)
    graph = build_rag_graph(services, config)

    result = await graph.ainvoke(
        _initial_state(query=raw_query)
        | {
            "raw_query": raw_query,
            "resolved_query": resolved_query,
        }
    )

    assert result["answer"] == _answer_with_footer(
        "GDP tăng 7%",
        title="Bonds",
        url="https://example.com/bonds",
        source="src.com",
        score=0.9,
    )
    services.guard.check_input.assert_awaited_once_with(resolved_query)
    services.embedder.embed_query.assert_awaited_once_with(resolved_query)
    services.reranker.rerank.assert_awaited_once()
    assert services.reranker.rerank.await_args.kwargs["query"] == resolved_query
    services.guard.check_output.assert_awaited_once_with(
        text="GDP tăng 7% [[cite:hybrid:1]]",
        prompt=resolved_query,
    )


@pytest.mark.asyncio
async def test_rag_pipeline_generation_prompt_uses_conversation_context_and_raw_query():
    from orchestrator.pipeline.rag import build_rag_graph

    retrieved = [{
        "id": "1",
        "text": "bond text",
        "source": "src.com",
        "title": "Bonds",
        "url": "https://example.com/bonds",
        "score": 0.7,
    }]
    reranked = [{"index": 0, "score": 0.9}]
    conversation_context = (
        "Tóm tắt hội thoại:\nNgười dùng đang hỏi về bất động sản và thị trường vốn."
    )

    services = _make_services(
        retrieved_docs=retrieved,
        reranked=reranked,
        llm_side_effect=["Answer with context [[cite:hybrid:1]]"],
    )
    config = _make_config(fallback_min_chunks=1)
    graph = build_rag_graph(services, config)

    result = await graph.ainvoke(
        _initial_state(query="Còn trái phiếu doanh nghiệp thì sao?")
        | {
            "raw_query": "Còn trái phiếu doanh nghiệp thì sao?",
            "resolved_query": "Trái phiếu doanh nghiệp ảnh hưởng thế nào đến thị trường vốn?",
            "conversation_context": conversation_context,
        }
    )

    assert result["answer"] == _answer_with_footer(
        "Answer with context",
        title="Bonds",
        url="https://example.com/bonds",
        source="src.com",
        score=0.9,
    )
    prompt = services.llm.generate.await_args.kwargs["user_prompt"]
    assert conversation_context in prompt
    assert "Còn trái phiếu doanh nghiệp thì sao?" in prompt
    assert "### Trả lời ngắn gọn" in prompt
    assert "----" in prompt


@pytest.mark.asyncio
async def test_rag_pipeline_preserves_web_provenance_in_final_context():
    from orchestrator.pipeline.rag import build_rag_graph

    retrieved = [{
        "id": "1",
        "text": "hybrid text",
        "source": "hybrid.local",
        "title": "Hybrid",
        "url": "https://example.com/hybrid",
        "score": 0.5,
    }]
    reranked = [{"index": 0, "score": 0.5}]
    web = [{
        "context_id": "web:0",
        "text": "web content",
        "source": "web.com",
        "source_type": "web",
        "retrieval_stage": "web_fallback",
        "original_rank": 0,
        "title": "Web",
        "url": "https://web.com/article",
        "score": 0.92,
        "collection_name": "",
        "doc_type": "web_page",
        "chunk_type": "web_snippet",
        "modality": "text",
        "source_quality": "external",
        "image_path": "",
        "structured_data": {},
    }]

    services = _make_services(
        retrieved_docs=retrieved,
        reranked=reranked,
        web_results=web,
        llm_side_effect=["Answer with web [[cite:web:0]]"],
    )
    config = _make_config(fallback_min_chunks=3)
    graph = build_rag_graph(services, config)

    result = await graph.ainvoke(_initial_state())

    assert any(item["source_type"] == "web" for item in result["final_context"])
    web_item = next(item for item in result["final_context"] if item["source_type"] == "web")
    assert web_item["context_id"] == "web:0"
    assert web_item["retrieval_stage"] == "web_fallback"


@pytest.mark.asyncio
async def test_rag_pipeline_builds_citations_from_provenance_pool_not_list_head():
    from orchestrator.pipeline.rag import build_rag_graph

    retrieved = [{
        "id": "1",
        "text": "hybrid text",
        "source": "hybrid.local",
        "title": "Hybrid",
        "url": "https://example.com/hybrid",
        "score": 0.55,
    }]
    reranked = [{"index": 0, "score": 0.55}]
    web = [{
        "context_id": "web:0",
        "text": "web content",
        "source": "web.com",
        "source_type": "web",
        "retrieval_stage": "web_fallback",
        "original_rank": 0,
        "title": "Web",
        "url": "https://web.com/article",
        "score": 0.95,
        "collection_name": "",
        "doc_type": "web_page",
        "chunk_type": "web_snippet",
        "modality": "text",
        "source_quality": "external",
        "image_path": "",
        "structured_data": {},
    }]

    services = _make_services(
        retrieved_docs=retrieved,
        reranked=reranked,
        web_results=web,
        llm_side_effect=["Answer with web [[cite:web:0]]"],
    )
    config = _make_config(fallback_min_chunks=3)
    config.rag.citation_limit = 1
    graph = build_rag_graph(services, config)

    result = await graph.ainvoke(_initial_state())

    assert "web:0" in result["citation_pool"]
    assert result["citations"][0]["context_id"] == "web:0"
    assert result["citations"][0]["source_type"] == "web"


@pytest.mark.asyncio
async def test_generation_prompt_contains_context_ids_and_source_metadata():
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
        llm_side_effect=["GDP tăng 7% [[cite:hybrid:1]]"],
    )
    config = _make_config(fallback_min_chunks=1)
    graph = build_rag_graph(services, config)

    await graph.ainvoke(_initial_state())

    prompt = services.llm.generate.await_args_list[0].kwargs["user_prompt"]
    assert "hybrid:1" in prompt
    assert "https://example.com/gdp" in prompt
    assert "Source:" in prompt
    assert "### Phân tích chính" in prompt


@pytest.mark.asyncio
async def test_generation_prompt_includes_web_context_when_fallback_runs():
    from orchestrator.pipeline.rag import build_rag_graph

    retrieved = [
        {
            "id": str(i),
            "text": f"internal text {i}",
            "source": f"internal{i}.local",
            "title": f"Internal {i}",
            "url": f"https://internal/{i}",
            "score": 0.55 - (i * 0.01),
        }
        for i in range(5)
    ]
    reranked = [{"index": i, "score": 0.55 - (i * 0.01)} for i in range(5)]
    web = [{
        "context_id": "web:0",
        "text": "Trường Đại học Thương mại là trường đại học công lập tại Hà Nội.",
        "source": "tmu.edu.vn",
        "title": "Giới thiệu chung về Trường Đại học Thương mại",
        "url": "https://tmu.edu.vn/trang/gioi-thieu-chung-ve-dai-hoc-thuong-mai",
        "score": 1.0,
    }]

    services = _make_services(
        retrieved_docs=retrieved,
        reranked=reranked,
        web_results=web,
        llm_side_effect=["Answer with web [[cite:web:0]]"],
    )
    config = _make_config(fallback_min_chunks=10)
    graph = build_rag_graph(services, config)

    await graph.ainvoke(_initial_state(query="cho tôi biết thông tin về trường đại học thương mại"))

    prompt = services.llm.generate.await_args_list[0].kwargs["user_prompt"]
    assert "https://tmu.edu.vn/trang/gioi-thieu-chung-ve-dai-hoc-thuong-mai" in prompt
    assert "Trường Đại học Thương mại là trường đại học công lập tại Hà Nội." in prompt


@pytest.mark.asyncio
async def test_rag_pipeline_text_mode_appends_structured_citation_footer():
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
        llm_side_effect=["GDP tăng 7% [[cite:hybrid:1]]"],
    )
    config = _make_config(fallback_min_chunks=1)
    graph = build_rag_graph(services, config)

    result = await graph.ainvoke(_initial_state())

    assert result["answer"] == _answer_with_footer(
        "GDP tăng 7%",
        title="GDP",
        url="https://example.com/gdp",
        source="src.com",
        score=0.9,
    )
    assert result["citations"][0]["context_id"] == "hybrid:1"


@pytest.mark.asyncio
async def test_rag_pipeline_text_mode_does_not_need_inline_citation_repair():
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

    assert services.llm.generate.await_count == 1
    assert result["answer"] == _answer_with_footer(
        "GDP tăng 7%",
        title="GDP",
        url="https://example.com/gdp",
        source="src.com",
        score=0.9,
    )


@pytest.mark.asyncio
async def test_rag_pipeline_text_mode_appends_web_citation_footer():
    from orchestrator.pipeline.rag import build_rag_graph

    retrieved = [{
        "id": "1",
        "text": "hybrid text",
        "source": "hybrid.local",
        "title": "Hybrid",
        "url": "https://example.com/hybrid",
        "score": 0.5,
    }]
    reranked = [{"index": 0, "score": 0.5}]
    web = [{
        "context_id": "web:0",
        "text": "web content",
        "source": "web.com",
        "source_type": "web",
        "retrieval_stage": "web_fallback",
        "original_rank": 0,
        "title": "Web",
        "url": "https://web.com/article",
        "score": 0.92,
        "collection_name": "",
        "doc_type": "web_page",
        "chunk_type": "web_snippet",
        "modality": "text",
        "source_quality": "external",
        "image_path": "",
        "structured_data": {},
    }]

    services = _make_services(
        retrieved_docs=retrieved,
        reranked=reranked,
        web_results=web,
        llm_side_effect=["Fed giữ nguyên lãi suất [[cite:web:0]]"],
    )
    config = _make_config(fallback_min_chunks=3)
    graph = build_rag_graph(services, config)

    result = await graph.ainvoke(_initial_state())

    assert result["answer"].startswith("Fed giữ nguyên lãi suất")
    assert "\n\n----\n\n### Nguồn trích dẫn\n" in result["answer"]
    assert "- [Web](https://web.com/article) - **web.com (0.9200)**" in result["answer"]
    assert len(result["citations"]) == 2
    assert result["citations"][0]["source_type"] == "web"


@pytest.mark.asyncio
async def test_rag_pipeline_audio_mode_uses_spoken_prompt_and_skips_footer():
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
        llm_side_effect=["Giá đang tăng, nhưng tăng chậm lại."],
    )
    config = _make_config(fallback_min_chunks=1)
    graph = build_rag_graph(services, config)

    result = await graph.ainvoke(
        _initial_state() | {"response_mode": "audio"}
    )

    prompt = services.llm.generate.await_args.kwargs["user_prompt"]
    assert "một đoạn ngắn" in prompt
    assert "văn nói tự nhiên" in prompt
    assert "### Trả lời ngắn gọn" not in prompt
    assert result["answer"] == "Giá đang tăng, nhưng tăng chậm lại."
    assert result["citations"][0]["context_id"] == "hybrid:1"


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
        llm_side_effect=["GDP tăng 7% [[cite:hybrid:1]]"],
    )
    config = _make_config(fallback_min_chunks=1)
    graph = build_rag_graph(services, config)

    result = await graph.ainvoke(_initial_state())

    assert result["input_safe"] is True
    assert result["answer"] == _answer_with_footer(
        "GDP tăng 7%",
        title="GDP",
        url="https://example.com/gdp",
        source="src.com",
        score=0.9,
    )
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
    """No context: empty retrieval + no web results -> neutral no_context_message, LLM not called."""
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
    assert "Xin lỗi" not in result["answer"]
    services.llm.generate.assert_not_called()


@pytest.mark.asyncio
async def test_rag_pipeline_web_fallback_triggered():
    """Web fallback: soft-threshold + shallow support triggers web search."""
    from orchestrator.pipeline.rag import build_rag_graph

    retrieved = [{
        "id": "1",
        "text": "t",
        "source": "s",
        "title": "T",
        "url": "https://example.com/t",
        "score": 0.8,
    }]
    reranked = [{"index": 0, "score": 0.8}]
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
        llm_side_effect=["Answer with web [[cite:web:0]]"],
    )
    config = _make_config(fallback_min_chunks=3)
    graph = build_rag_graph(services, config)

    result = await graph.ainvoke(_initial_state())

    services.web_search.search.assert_called_once()
    assert any(c["source"] == "web.com" for c in result["final_context"])
    assert result["answer"].startswith("Answer with web")
    assert "\n\n----\n\n### Nguồn trích dẫn\n" in result["answer"]
    assert "- [Web](https://web.com/article) - **web.com (0.4200)**" in result["answer"]


@pytest.mark.asyncio
async def test_rag_pipeline_web_fallback_uses_hard_threshold():
    from orchestrator.pipeline.rag import build_rag_graph

    retrieved = [
        {
            "id": "1",
            "text": "GDP text",
            "source": "src.com",
            "title": "GDP",
            "url": "https://example.com/gdp",
            "score": 0.69,
        },
        {
            "id": "2",
            "text": "supporting text",
            "source": "src.com",
            "title": "GDP support",
            "url": "https://example.com/gdp-support",
            "score": 0.6,
        },
    ]
    reranked = [{"index": 0, "score": 0.69}, {"index": 1, "score": 0.68}]
    web = [
        {
            "text": "web content",
            "source": "web.com",
            "title": "Web",
            "url": "https://web.com/article",
            "score": 0.88,
        }
    ]

    services = _make_services(
        retrieved_docs=retrieved,
        reranked=reranked,
        web_results=web,
        llm_side_effect=["Answer with web [[cite:web:0]]"],
    )
    config = _make_config(
        fallback_min_chunks=1,
        web_fallback_min_chunks=2,
        web_fallback_hard_threshold=0.7,
        web_fallback_soft_threshold=0.85,
    )
    graph = build_rag_graph(services, config)

    result = await graph.ainvoke(_initial_state())

    services.web_search.search.assert_awaited_once_with("GDP Việt Nam?")
    assert any(item["source_type"] == "web" for item in result["final_context"])


@pytest.mark.asyncio
async def test_rag_pipeline_web_fallback_uses_soft_threshold_when_support_is_shallow():
    from orchestrator.pipeline.rag import build_rag_graph

    retrieved = [
        {
            "id": "1",
            "text": "GDP text",
            "source": "src.com",
            "title": "GDP",
            "url": "https://example.com/gdp",
            "score": 0.8,
        }
    ]
    reranked = [{"index": 0, "score": 0.8}]
    web = [
        {
            "text": "web content",
            "source": "web.com",
            "title": "Web",
            "url": "https://web.com/article",
            "score": 0.88,
        }
    ]

    services = _make_services(
        retrieved_docs=retrieved,
        reranked=reranked,
        web_results=web,
        llm_side_effect=["Answer with web [[cite:web:0]]"],
    )
    config = _make_config(
        fallback_min_chunks=1,
        web_fallback_min_chunks=2,
        web_fallback_hard_threshold=0.7,
        web_fallback_soft_threshold=0.85,
    )
    graph = build_rag_graph(services, config)

    await graph.ainvoke(_initial_state())

    services.web_search.search.assert_awaited_once_with("GDP Việt Nam?")


@pytest.mark.asyncio
async def test_rag_pipeline_skips_web_fallback_when_internal_evidence_is_strong():
    from orchestrator.pipeline.rag import build_rag_graph

    retrieved = [
        {
            "id": "1",
            "text": "GDP text",
            "source": "src.com",
            "title": "GDP",
            "url": "https://example.com/gdp",
            "score": 0.9,
        },
        {
            "id": "2",
            "text": "supporting text",
            "source": "src.com",
            "title": "GDP support",
            "url": "https://example.com/gdp-support",
            "score": 0.87,
        },
    ]
    reranked = [{"index": 0, "score": 0.9}, {"index": 1, "score": 0.87}]

    services = _make_services(
        retrieved_docs=retrieved,
        reranked=reranked,
        web_results=[
            {
                "text": "web content",
                "source": "web.com",
                "title": "Web",
                "url": "https://web.com/article",
                "score": 0.88,
            }
        ],
        llm_side_effect=["Answer with docs [[cite:hybrid:1]]"],
    )
    config = _make_config(
        fallback_min_chunks=1,
        web_fallback_min_chunks=2,
        web_fallback_hard_threshold=0.7,
        web_fallback_soft_threshold=0.85,
    )
    graph = build_rag_graph(services, config)

    result = await graph.ainvoke(_initial_state())

    services.web_search.search.assert_not_called()
    assert all(item["source_type"] != "web" for item in result["final_context"])


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
        llm_side_effect=[
            "Unsafe generated text",
            "Safe regenerated text [[cite:hybrid:1]]",
        ],
        output_guard_side_effect=[UNSAFE_VIOLENT_RESULT, SAFE_OUTPUT_RESULT],
    )
    config = _make_config(fallback_min_chunks=1)
    graph = build_rag_graph(services, config)

    result = await graph.ainvoke(_initial_state())

    assert result["output_safe"] is True
    assert result["answer"] == _answer_with_footer(
        "Safe regenerated text",
        title="T",
        url="https://example.com/t",
        source="s",
        score=0.9,
    )
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
