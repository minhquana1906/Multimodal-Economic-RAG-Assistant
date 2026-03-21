from __future__ import annotations

import json
import pytest
from unittest.mock import AsyncMock, MagicMock
from httpx import AsyncClient, ASGITransport
from fastapi import FastAPI

from orchestrator.routers.chat import create_chat_router
from orchestrator.pipeline.rag import RAGState


def _make_app(rag_result: dict) -> FastAPI:
    """Build a minimal FastAPI app with a mocked RAG graph."""
    mock_graph = MagicMock()
    mock_graph.ainvoke = AsyncMock(return_value=rag_result)

    app = FastAPI()
    router = create_chat_router(mock_graph)
    app.include_router(router)
    return app


@pytest.mark.asyncio
async def test_chat_endpoint_non_streaming():
    """Non-streaming: returns valid ChatResponse with answer."""
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
    assert "choices" in data
    assert data["choices"][0]["delta"]["content"] == "GDP tăng 7%"
    assert data["choices"][0]["delta"]["role"] == "assistant"


@pytest.mark.asyncio
async def test_chat_endpoint_streaming():
    """Streaming: returns SSE events with answer chunks and [DONE] marker."""
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
                    chunks.append(line[6:])  # strip "data: " prefix

    assert "[DONE]" in chunks
    # At least one data chunk before [DONE]
    data_chunks = [c for c in chunks if c != "[DONE]"]
    assert len(data_chunks) >= 1
    # Each non-DONE chunk is valid JSON
    for chunk in data_chunks:
        parsed = json.loads(chunk)
        assert "delta" in parsed
