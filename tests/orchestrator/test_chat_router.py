from __future__ import annotations

import json
import pytest
from unittest.mock import AsyncMock, MagicMock
from httpx import AsyncClient, ASGITransport
from fastapi import FastAPI

from orchestrator.routers.chat import create_chat_router


def _make_app(rag_result: dict) -> FastAPI:
    """Build a minimal FastAPI app with a mocked RAG graph."""
    mock_graph = MagicMock()
    mock_graph.ainvoke = AsyncMock(return_value=rag_result)
    app = FastAPI()
    app.include_router(create_chat_router(mock_graph))
    return app


@pytest.mark.asyncio
async def test_chat_endpoint_non_streaming():
    """Non-streaming: returns valid ChatResponse with answer and correct object type."""
    app = _make_app({
        "answer": "GDP tăng 7%",
        "citations": [{"title": "Báo cáo GDP", "source": "mof.gov.vn"}],
    })

    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as client:
        response = await client.post(
            "/v1/chat/completions",
            json={
                "model": "multimodal-rag",
                "messages": [{"role": "user", "content": "GDP Việt Nam?"}],
                "stream": False,
            },
        )

    assert response.status_code == 200
    data = response.json()
    assert data["object"] == "chat.completion"
    assert "choices" in data
    assert data["choices"][0]["delta"]["content"] == "GDP tăng 7%"
    assert data["choices"][0]["delta"]["role"] == "assistant"


@pytest.mark.asyncio
async def test_chat_endpoint_streaming():
    """Streaming: returns SSE with OpenAI-format chunks containing top-level choices array."""
    app = _make_app({
        "answer": "GDP tăng",
        "citations": [],
    })

    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as client:
        async with client.stream(
            "POST",
            "/v1/chat/completions",
            json={
                "model": "multimodal-rag",
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

    # Each data chunk must be a valid OpenAI stream chunk with top-level choices
    for chunk in data_chunks:
        parsed = json.loads(chunk)
        assert "choices" in parsed
        assert "object" in parsed
        assert parsed["object"] == "chat.completion.chunk"
        assert "delta" in parsed["choices"][0]

    # Verify content comes from this test's mock
    all_content = "".join(
        parsed["choices"][0]["delta"].get("content", "")
        for c in data_chunks
        if (parsed := json.loads(c))
    )
    assert "GDP" in all_content

    # At least one chunk must have finish_reason="stop"
    has_stop = any(
        json.loads(c)["choices"][0].get("finish_reason") == "stop"
        for c in data_chunks
    )
    assert has_stop, "No chunk with finish_reason='stop' found"


@pytest.mark.asyncio
async def test_chat_endpoint_no_user_message_returns_400():
    """Returns 400 when messages contains no user-role message."""
    app = _make_app({"answer": "", "citations": []})

    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as client:
        response = await client.post(
            "/v1/chat/completions",
            json={
                "model": "multimodal-rag",
                "messages": [{"role": "system", "content": "You are helpful."}],
                "stream": False,
            },
        )

    assert response.status_code == 400
    assert "user" in response.json()["detail"].lower()
