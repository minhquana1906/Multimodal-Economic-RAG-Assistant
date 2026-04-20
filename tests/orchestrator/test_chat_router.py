from __future__ import annotations

import json
import re
from unittest.mock import AsyncMock, MagicMock

import pytest
from fastapi import FastAPI
from httpx import ASGITransport, AsyncClient

from orchestrator.config import PromptsConfig
from orchestrator.routers.chat import create_chat_router, execute_chat_turn
from orchestrator.models.schemas import Message


def _make_app(
    rag_result: dict,
    task_result: str = "task result",
    general_result: str = "general result",
    stream_tokens: list[str] | None = None,
    detect_intent_route: str = "rag",
) -> tuple[FastAPI, MagicMock, MagicMock, PromptsConfig]:
    """Build a minimal FastAPI app with mocked RAG, auxiliary, and general chat handlers."""
    mock_graph = MagicMock()
    mock_graph.ainvoke = AsyncMock(return_value=rag_result)
    mock_task_llm = MagicMock()
    prompts = PromptsConfig()

    async def _complete_prompt(prompt: str, max_tokens: int | None = None) -> str:
        del max_tokens
        if "Bạn là bộ chuẩn hóa truy vấn" in prompt:
            match = re.search(r"Câu hỏi gốc:\n(?P<query>.+)", prompt)
            return match.group("query").strip() if match else "GDP Việt Nam?"
        if "Chỉ trả về đúng một nhãn" in prompt:
            lowered = prompt.lower()
            if "lời chào" in lowered or "xin chào" in lowered:
                return "general_chat"
            return "rag"
        return task_result

    mock_task_llm.complete_prompt = AsyncMock(side_effect=_complete_prompt)
    mock_task_llm.generate = AsyncMock(return_value=general_result)

    # detect_intent uses the new routing flow
    last_user_content = rag_result.get("_test_resolved_query", "")

    async def _detect_intent(*, system_prompt, user_prompt, fallback_query):
        del system_prompt, user_prompt
        return {
            "route": detect_intent_route,
            "resolved_query": fallback_query or last_user_content,
        }

    mock_task_llm.detect_intent = AsyncMock(side_effect=_detect_intent)

    # stream_chat yields provided tokens or a single "stream result" token
    _tokens = stream_tokens if stream_tokens is not None else ["stream result"]

    async def _stream_chat(messages):
        for token in _tokens:
            yield token

    mock_task_llm.stream_chat = _stream_chat

    # Retrieval-only graph mock: returns empty context (streaming path skips LLM)
    mock_retrieval_graph = MagicMock()
    mock_retrieval_graph.ainvoke = AsyncMock(
        return_value={
            "final_context": [],
            "citation_pool": {},
            "embeddings": [],
            "retrieved_docs": [],
            "reranked_docs": [],
            "web_results": [],
            "error": None,
        }
    )

    app = FastAPI()
    app.include_router(
        create_chat_router(
            mock_graph,
            retrieval_graph=mock_retrieval_graph,
            task_llm=mock_task_llm,
            prompts=prompts,
        )
    )
    return app, mock_graph, mock_task_llm, prompts


@pytest.mark.asyncio
async def test_chat_endpoint_non_streaming():
    """Non-streaming: returns canonical message.content payload."""
    app, mock_graph, mock_task_llm, _ = _make_app(
        {
            "answer": "GDP tăng 7%",
            "citations": [
                {
                    "title": "Báo cáo GDP",
                    "url": "https://mof.gov.vn/gdp",
                    "source": "mof.gov.vn",
                    "score": 0.9123,
                }
            ],
        }
    )

    async with AsyncClient(
        transport=ASGITransport(app=app), base_url="http://test"
    ) as client:
        response = await client.post(
            "/v1/chat/completions",
            json={
                "model": "multimodal-economic-rag",
                "messages": [{"role": "user", "content": "GDP Việt Nam?"}],
                "stream": False,
            },
        )

    assert response.status_code == 200
    data = response.json()
    assert data["object"] == "chat.completion"
    assert "choices" in data
    assert data["choices"][0]["message"]["content"] == "GDP tăng 7%"
    assert data["choices"][0]["message"]["role"] == "assistant"
    mock_graph.ainvoke.assert_awaited_once()
    mock_task_llm.detect_intent.assert_awaited_once()


@pytest.mark.asyncio
async def test_chat_endpoint_streaming():
    """Streaming: continues to emit delta.content chunks via stream_chat."""
    app, _, _, _ = _make_app(
        {
            "answer": "GDP tăng",
            "citations": [],
        },
        stream_tokens=["GDP", " tăng"],
    )

    async with AsyncClient(
        transport=ASGITransport(app=app), base_url="http://test"
    ) as client:
        async with client.stream(
            "POST",
            "/v1/chat/completions",
            json={
                "model": "multimodal-economic-rag",
                "messages": [{"role": "user", "content": "GDP?"}],
                "stream": True,
            },
        ) as response:
            assert response.status_code == 200
            assert "text/event-stream" in response.headers["content-type"]

            chunks = []
            async for line in response.aiter_lines():
                if line.startswith("data: "):
                    chunks.append(line[6:])

    assert "[DONE]" in chunks
    data_chunks = [c for c in chunks if c != "[DONE]"]
    assert len(data_chunks) >= 1

    for chunk in data_chunks:
        parsed = json.loads(chunk)
        assert "choices" in parsed
        assert "object" in parsed
        assert parsed["object"] == "chat.completion.chunk"
        assert "delta" in parsed["choices"][0]

    all_content = "".join(
        parsed["choices"][0]["delta"].get("content", "")
        for c in data_chunks
        if (parsed := json.loads(c))
    )
    assert "GDP" in all_content

    has_stop = any(
        json.loads(c)["choices"][0].get("finish_reason") == "stop"
        for c in data_chunks
    )
    assert has_stop, "No chunk with finish_reason='stop' found"


@pytest.mark.asyncio
async def test_chat_endpoint_streaming_preserves_inline_markdown_answer():
    """Streaming via stream_chat relays tokens without adding citation footer."""
    expected = "GDP tăng theo [Báo cáo GDP](https://mof.gov.vn/gdp)\n\n- Mục 1\n- **Mục 2**"
    app, _, _, _ = _make_app(
        {
            "answer": expected,
            "citations": [
                {
                    "title": "Báo cáo GDP",
                    "url": "https://mof.gov.vn/gdp",
                    "source": "mof.gov.vn",
                    "score": 0.9123,
                }
            ],
        },
        stream_tokens=["GDP tăng theo [Báo cáo GDP](https://mof.gov.vn/gdp)", "\n\n- Mục 1\n- **Mục 2**"],
    )

    async with AsyncClient(
        transport=ASGITransport(app=app), base_url="http://test"
    ) as client:
        async with client.stream(
            "POST",
            "/v1/chat/completions",
            json={
                "model": "multimodal-economic-rag",
                "messages": [{"role": "user", "content": "GDP?"}],
                "stream": True,
            },
        ) as response:
            chunks = []
            async for line in response.aiter_lines():
                if line.startswith("data: "):
                    chunks.append(line[6:])

    data_chunks = [json.loads(c) for c in chunks if c != "[DONE]"]
    all_content = "".join(
        chunk["choices"][0]["delta"].get("content", "") for chunk in data_chunks
    )
    assert all_content == expected
    assert "**Nguồn:**" not in all_content


@pytest.mark.asyncio
async def test_chat_endpoint_streaming_matches_non_streaming_multiline_markdown():
    """Non-streaming returns the graph answer; streaming relays stream_chat tokens."""
    answer = "GDP tăng theo [Báo cáo GDP](https://mof.gov.vn/gdp)\n\n- Mục 1\n- **Mục 2**"
    app, _, _, _ = _make_app(
        {
            "answer": answer,
            "citations": [
                {
                    "title": "Báo cáo GDP",
                    "url": "https://mof.gov.vn/gdp",
                    "source": "mof.gov.vn",
                    "score": 0.9123,
                }
            ],
        },
        stream_tokens=["GDP tăng theo [Báo cáo GDP](https://mof.gov.vn/gdp)", "\n\n- Mục 1\n- **Mục 2**"],
    )

    async with AsyncClient(
        transport=ASGITransport(app=app), base_url="http://test"
    ) as client:
        non_streaming = await client.post(
            "/v1/chat/completions",
            json={
                "model": "multimodal-economic-rag",
                "messages": [{"role": "user", "content": "GDP?"}],
                "stream": False,
            },
        )
        async with client.stream(
            "POST",
            "/v1/chat/completions",
            json={
                "model": "multimodal-economic-rag",
                "messages": [{"role": "user", "content": "GDP?"}],
                "stream": True,
            },
        ) as response:
            chunks = []
            async for line in response.aiter_lines():
                if line.startswith("data: "):
                    chunks.append(line[6:])

    stream_chunks = [json.loads(c) for c in chunks if c != "[DONE]"]
    streamed = "".join(chunk["choices"][0]["delta"].get("content", "") for chunk in stream_chunks)
    assert non_streaming.json()["choices"][0]["message"]["content"] == answer
    assert streamed == answer


@pytest.mark.asyncio
async def test_chat_endpoint_no_user_message_returns_400():
    """Returns 400 when messages contains no user-role message."""
    app, _, _, _ = _make_app({"answer": "", "citations": []})

    async with AsyncClient(
        transport=ASGITransport(app=app), base_url="http://test"
    ) as client:
        response = await client.post(
            "/v1/chat/completions",
            json={
                "model": "multimodal-economic-rag",
                "messages": [{"role": "system", "content": "You are helpful."}],
                "stream": False,
            },
        )

    assert response.status_code == 400
    assert "user" in response.json()["detail"].lower()


@pytest.mark.asyncio
async def test_execute_chat_turn_returns_compact_trace_output():
    mock_graph = MagicMock()
    mock_graph.ainvoke = AsyncMock(
        return_value={
            "answer": "GDP tăng 7% [GDP](https://example.com/gdp)",
            "citations": [{"context_id": "hybrid:1"}],
            "generation_prompt": "should not leak",
            "final_context": [{"context_id": "hybrid:1"}],
        }
    )

    result = await execute_chat_turn(
        mock_graph,
        None,
        messages=[{"role": "user", "content": "GDP Việt Nam?"}],
        max_tokens=None,
    )

    assert result["answer"] == "GDP tăng 7% [GDP](https://example.com/gdp)"
    assert result["citations"] == [{"context_id": "hybrid:1"}]
    assert result["task_type"] == "rag"
    assert result["resolved_query"] == "GDP Việt Nam?"
    # Internal fields must not surface in the API result
    assert "generation_prompt" not in result
    assert "final_context" not in result


@pytest.mark.asyncio
async def test_streaming_relays_upstream_deltas_without_rechunking():
    """Streaming path calls stream_chat() and relays each delta token immediately."""
    tokens = ["GDP", " tăng", " 6.93%", " [S1]"]
    app, _, mock_task_llm, _ = _make_app(
        {
            "answer": "GDP tăng 6.93% [S1]",
            "citations": [],
            "generation_prompt": "context prompt",
        },
        stream_tokens=tokens,
    )

    async with AsyncClient(
        transport=ASGITransport(app=app), base_url="http://test"
    ) as client:
        async with client.stream(
            "POST",
            "/v1/chat/completions",
            json={
                "model": "multimodal-economic-rag",
                "messages": [{"role": "user", "content": "GDP Việt Nam 2024?"}],
                "stream": True,
            },
        ) as response:
            assert response.status_code == 200
            assert "text/event-stream" in response.headers["content-type"]

            received = []
            async for line in response.aiter_lines():
                if line.startswith("data: ") and line[6:] != "[DONE]":
                    chunk = json.loads(line[6:])
                    content = chunk["choices"][0]["delta"].get("content")
                    if content:
                        received.append(content)

    # Each token arrives as its own SSE event (no rechunking)
    assert received == tokens
    all_text = "".join(received)
    assert "GDP" in all_text
    assert "[S1]" in all_text


def test_rag_prompt_uses_inline_source_ids():
    """build_rag_prompt formats sources with [S1], [S2] IDs and citation instructions."""
    from orchestrator.pipeline.rag_prompts import build_rag_prompt

    sources = [
        {
            "source_id": "S1",
            "title": "Báo cáo GDP",
            "url": "https://gso.gov.vn/gdp",
            "text": "GDP tăng 6.93% năm 2023.",
        },
        {
            "source_id": "S2",
            "title": "World Bank",
            "url": "https://worldbank.org/vn",
            "text": "Tăng trưởng ổn định.",
        },
    ]
    prompt = build_rag_prompt(sources)

    assert "[S1]" in prompt
    assert "[S2]" in prompt
    assert "Báo cáo GDP" in prompt
    assert "https://gso.gov.vn/gdp" in prompt
    assert "GDP tăng 6.93%" in prompt
    assert "Không tạo nguồn ngoài danh sách" in prompt
