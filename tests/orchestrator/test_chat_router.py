from __future__ import annotations

import json
from unittest.mock import AsyncMock, MagicMock

import pytest
from fastapi import FastAPI
from httpx import ASGITransport, AsyncClient

from orchestrator.routers.chat import create_chat_router, execute_chat_turn


def _make_app(
    rag_result: dict,
    task_result: str = "task result",
) -> tuple[FastAPI, MagicMock, MagicMock]:
    """Build a minimal FastAPI app with mocked RAG and auxiliary task handlers."""
    mock_graph = MagicMock()
    mock_graph.ainvoke = AsyncMock(return_value=rag_result)
    mock_task_llm = MagicMock()
    mock_task_llm.complete_prompt = AsyncMock(return_value=task_result)

    app = FastAPI()
    app.include_router(create_chat_router(mock_graph, mock_task_llm))
    return app, mock_graph, mock_task_llm


@pytest.mark.asyncio
async def test_chat_endpoint_non_streaming():
    """Non-streaming: returns canonical message.content payload."""
    app, mock_graph, mock_task_llm = _make_app(
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
    mock_task_llm.complete_prompt.assert_not_called()


@pytest.mark.asyncio
async def test_chat_endpoint_streaming():
    """Streaming: continues to emit delta.content chunks."""
    app, _, _ = _make_app(
        {
            "answer": "GDP tăng",
            "citations": [],
        }
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
async def test_chat_endpoint_builds_backend_conversation_state():
    app, mock_graph, _ = _make_app(
        {"answer": "Trái phiếu chịu áp lực", "citations": []}
    )

    async with AsyncClient(
        transport=ASGITransport(app=app), base_url="http://test"
    ) as client:
        response = await client.post(
            "/v1/chat/completions",
            json={
                "model": "multimodal-economic-rag",
                "messages": [
                    {"role": "system", "content": "You are helpful."},
                    {
                        "role": "user",
                        "content": "Bất động sản đang ảnh hưởng thế nào đến thị trường vốn?",
                    },
                    {"role": "assistant", "content": "Áp lực vốn vẫn còn cao."},
                    {"role": "user", "content": "Còn trái phiếu doanh nghiệp thì sao?"},
                ],
                "stream": False,
            },
        )

    assert response.status_code == 200
    state = mock_graph.ainvoke.await_args.args[0]
    assert state["raw_query"] == "Còn trái phiếu doanh nghiệp thì sao?"
    assert state["task_type"] == "chat"
    assert "Bất động sản đang ảnh hưởng" in state["conversation_context"]
    assert "Còn trái phiếu doanh nghiệp thì sao?" in state["resolved_query"]
    assert state["response_mode"] == "text"


@pytest.mark.asyncio
async def test_chat_endpoint_passes_audio_response_mode_into_state():
    app, mock_graph, _ = _make_app(
        {"answer": "Giá đang tăng chậm lại.", "citations": []}
    )

    async with AsyncClient(
        transport=ASGITransport(app=app), base_url="http://test"
    ) as client:
        response = await client.post(
            "/v1/chat/completions",
            json={
                "model": "multimodal-economic-rag",
                "messages": [{"role": "user", "content": "Giá nhà đang tăng hay giảm?"}],
                "response_mode": "audio",
                "stream": False,
            },
        )

    assert response.status_code == 200
    state = mock_graph.ainvoke.await_args.args[0]
    assert state["response_mode"] == "audio"


@pytest.mark.asyncio
async def test_chat_endpoint_streaming_preserves_inline_markdown_answer():
    app, _, _ = _make_app(
        {
            "answer": "GDP tăng theo [Báo cáo GDP](https://mof.gov.vn/gdp)\n\n- Mục 1\n- **Mục 2**",
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
    assert all_content == (
        "GDP tăng theo [Báo cáo GDP](https://mof.gov.vn/gdp)\n\n- Mục 1\n- **Mục 2**"
    )
    assert "**Nguồn:**" not in all_content


@pytest.mark.asyncio
async def test_chat_endpoint_streaming_matches_non_streaming_multiline_markdown():
    answer = "GDP tăng theo [Báo cáo GDP](https://mof.gov.vn/gdp)\n\n- Mục 1\n- **Mục 2**"
    app, _, _ = _make_app(
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
        }
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
    app, _, _ = _make_app({"answer": "", "citations": []})

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
@pytest.mark.parametrize(
    ("task_prompt", "task_result"),
    [
        (
            """### Task:
Generate a concise, 3-5 word title with an emoji summarizing the chat history.

### Chat History:
<chat_history>
USER: bất động sản ảnh hưởng như nào đến thị trường vốn
ASSISTANT: ...
</chat_history>""",
            "Thi truong von",
        ),
        (
            """### Task:
Generate 1-3 broad tags categorizing the main themes of the chat history, along with 1-3 more specific subtopic tags.

### Chat History:
<chat_history>
USER: bất động sản ảnh hưởng như nào đến thị trường vốn
ASSISTANT: ...
</chat_history>""",
            '{"tags": ["Business", "Real Estate"]}',
        ),
        (
            """### Task:
Suggest 3-5 relevant follow-up questions or prompts based on the chat history.

### Chat History:
<chat_history>
USER: bất động sản ảnh hưởng như nào đến thị trường vốn
ASSISTANT: ...
</chat_history>""",
            '{"follow_ups": ["Lãi suất tác động ra sao?"]}',
        ),
    ],
)
async def test_chat_endpoint_bypasses_rag_for_auxiliary_tasks(
    task_prompt: str,
    task_result: str,
):
    app, mock_graph, mock_task_llm = _make_app(
        {"answer": "should not be used", "citations": []},
        task_result=task_result,
    )

    async with AsyncClient(
        transport=ASGITransport(app=app), base_url="http://test"
    ) as client:
        response = await client.post(
            "/v1/chat/completions",
            json={
                "model": "multimodal-economic-rag",
                "messages": [{"role": "user", "content": task_prompt}],
                "stream": False,
            },
        )

    assert response.status_code == 200
    data = response.json()
    assert data["choices"][0]["message"]["content"] == task_result
    mock_graph.ainvoke.assert_not_called()
    mock_task_llm.complete_prompt.assert_awaited_once()


@pytest.mark.asyncio
async def test_chat_endpoint_builds_auxiliary_history_from_messages():
    app, mock_graph, mock_task_llm = _make_app(
        {"answer": "should not be used", "citations": []},
        task_result="Thi truong von",
    )
    task_prompt = (
        "### Task:\n"
        "Generate a concise, 3-5 word title with an emoji summarizing the chat history."
    )

    async with AsyncClient(
        transport=ASGITransport(app=app), base_url="http://test"
    ) as client:
        response = await client.post(
            "/v1/chat/completions",
            json={
                "model": "multimodal-economic-rag",
                "messages": [
                    {
                        "role": "user",
                        "content": "bất động sản ảnh hưởng như nào đến thị trường vốn",
                    },
                    {"role": "assistant", "content": "..."},
                    {"role": "user", "content": task_prompt},
                ],
                "stream": False,
            },
        )

    assert response.status_code == 200
    prompt = mock_task_llm.complete_prompt.await_args.args[0]
    assert "<chat_history>" in prompt
    assert "USER: bất động sản ảnh hưởng như nào đến thị trường vốn" in prompt
    assert "ASSISTANT: ..." in prompt
    mock_graph.ainvoke.assert_not_called()


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

    assert result == {
        "answer": "GDP tăng 7% [GDP](https://example.com/gdp)",
        "citations": [{"context_id": "hybrid:1"}],
        "task_type": "chat",
        "resolved_query": "GDP Việt Nam?",
    }
