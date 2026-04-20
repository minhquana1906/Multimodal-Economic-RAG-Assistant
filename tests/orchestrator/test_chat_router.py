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
) -> tuple[FastAPI, MagicMock, MagicMock, MagicMock, PromptsConfig]:
    """Build a minimal FastAPI app with mocked RAG, auxiliary, and general chat handlers."""
    mock_graph = MagicMock()
    mock_graph.ainvoke = AsyncMock(return_value=rag_result)
    mock_task_llm = MagicMock()
    mock_guard = MagicMock()
    mock_guard.check_input = AsyncMock(
        return_value={"label": "safe", "safe_label": "Safe", "categories": [], "refusal": None}
    )
    mock_guard.check_output = AsyncMock(
        return_value={"label": "safe", "safe_label": "Safe", "categories": [], "refusal": "No"}
    )
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

    async def _detect_intent(messages):
        last = next(
            (m.get("content", "") for m in reversed(messages) if m.get("role") == "user"),
            last_user_content,
        )
        return {"route": detect_intent_route, "resolved_query": last}

    mock_task_llm.detect_intent = AsyncMock(side_effect=_detect_intent)

    # stream_chat yields provided tokens or a single "stream result" token
    _tokens = stream_tokens if stream_tokens is not None else ["stream result"]

    async def _stream_chat(messages):
        for token in _tokens:
            yield token

    mock_task_llm.stream_chat = _stream_chat

    app = FastAPI()
    app.include_router(create_chat_router(mock_graph, mock_task_llm, mock_guard, prompts))
    return app, mock_graph, mock_task_llm, mock_guard, prompts


@pytest.mark.asyncio
async def test_chat_endpoint_non_streaming():
    """Non-streaming: returns canonical message.content payload."""
    app, mock_graph, mock_task_llm, _, _ = _make_app(
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
    app, _, _, _, _ = _make_app(
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
async def test_chat_endpoint_builds_backend_conversation_state():
    app, mock_graph, _, _, _ = _make_app(
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
    assert state["task_type"] == "rag"
    assert "Bất động sản đang ảnh hưởng" in state["conversation_context"]
    assert "Còn trái phiếu doanh nghiệp thì sao?" in state["resolved_query"]
    assert state["response_mode"] == "text"


@pytest.mark.asyncio
async def test_chat_endpoint_passes_audio_response_mode_into_state():
    app, mock_graph, _, _, _ = _make_app(
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
async def test_chat_endpoint_infers_audio_response_mode_from_modalities():
    app, mock_graph, _, _, _ = _make_app(
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
                "modalities": ["text", "audio"],
                "audio": {"voice": "alloy", "format": "wav"},
                "stream": False,
            },
        )

    assert response.status_code == 200
    state = mock_graph.ainvoke.await_args.args[0]
    assert state["response_mode"] == "audio"


@pytest.mark.asyncio
async def test_chat_endpoint_streaming_preserves_inline_markdown_answer():
    """Streaming via stream_chat relays tokens without adding citation footer."""
    expected = "GDP tăng theo [Báo cáo GDP](https://mof.gov.vn/gdp)\n\n- Mục 1\n- **Mục 2**"
    app, _, _, _, _ = _make_app(
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
    app, _, _, _, _ = _make_app(
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
    app, _, _, _, _ = _make_app({"answer": "", "citations": []})

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
    app, mock_graph, mock_task_llm, _, _ = _make_app(
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
    app, mock_graph, mock_task_llm, _, _ = _make_app(
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
async def test_chat_endpoint_bypasses_rag_for_general_chat():
    app, mock_graph, mock_task_llm, mock_guard, _ = _make_app(
        {"answer": "should not be used", "citations": []},
        general_result="Mình có thể giúp bạn viết lại lời chào theo giọng lịch sự hơn.",
    )

    async with AsyncClient(
        transport=ASGITransport(app=app), base_url="http://test"
    ) as client:
        response = await client.post(
            "/v1/chat/completions",
            json={
                "model": "multimodal-economic-rag",
                "messages": [
                    {"role": "user", "content": "Bạn có thể viết lại lời chào này cho lịch sự hơn không?"}
                ],
                "stream": False,
            },
        )

    assert response.status_code == 200
    assert response.json()["choices"][0]["message"]["content"] == (
        "Mình có thể giúp bạn viết lại lời chào theo giọng lịch sự hơn."
    )
    mock_graph.ainvoke.assert_not_called()
    mock_guard.check_input.assert_awaited_once()
    mock_guard.check_output.assert_awaited_once()
    mock_task_llm.generate.assert_awaited_once()


@pytest.mark.asyncio
async def test_chat_endpoint_returns_safe_redirect_for_general_chat_live_facts():
    app, mock_graph, mock_task_llm, mock_guard, prompts = _make_app(
        {"answer": "should not be used", "citations": []},
        general_result="should not be used",
    )

    async with AsyncClient(
        transport=ASGITransport(app=app), base_url="http://test"
    ) as client:
        response = await client.post(
            "/v1/chat/completions",
            json={
                "model": "multimodal-economic-rag",
                "messages": [{"role": "user", "content": "Thời tiết hôm nay thế nào?"}],
                "stream": False,
            },
        )

    assert response.status_code == 200
    assert response.json()["choices"][0]["message"]["content"] == prompts.general_chat_live_facts_message
    mock_graph.ainvoke.assert_not_called()
    mock_guard.check_input.assert_awaited_once()
    mock_task_llm.generate.assert_not_awaited()


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
async def test_execute_chat_turn_rewrites_every_query_before_graph_call():
    mock_graph = MagicMock()
    mock_graph.ainvoke = AsyncMock(return_value={"answer": "ok", "citations": []})
    mock_task_llm = MagicMock()
    mock_task_llm.complete_prompt = AsyncMock(
        return_value="Trái phiếu doanh nghiệp tại Việt Nam đang chịu tác động thế nào?"
    )

    result = await execute_chat_turn(
        mock_graph,
        mock_task_llm,
        messages=[Message(role="user", content="còn trái phiếu dn thì sao")],
        max_tokens=None,
    )

    state = mock_graph.ainvoke.await_args.args[0]
    assert (
        state["resolved_query"]
        == "Trái phiếu doanh nghiệp tại Việt Nam đang chịu tác động thế nào?"
    )
    assert result["resolved_query"] == state["resolved_query"]


@pytest.mark.asyncio
async def test_execute_chat_turn_uses_rewritten_query_to_route_general_chat():
    mock_graph = MagicMock()
    mock_graph.ainvoke = AsyncMock(return_value={"answer": "should not be used", "citations": []})
    mock_task_llm = MagicMock()
    mock_guard = MagicMock()
    mock_guard.check_input = AsyncMock(
        return_value={"label": "safe", "safe_label": "Safe", "categories": [], "refusal": None}
    )
    mock_guard.check_output = AsyncMock(
        return_value={"label": "safe", "safe_label": "Safe", "categories": [], "refusal": "No"}
    )
    prompts = PromptsConfig()

    async def _complete_prompt(prompt: str, max_tokens: int | None = None) -> str:
        del max_tokens
        if "Bạn là bộ chuẩn hóa truy vấn" in prompt:
            return "Bạn có thể giúp mình viết lại một lời chào khách hàng theo giọng lịch sự hơn được không?"
        return "rag"

    mock_task_llm.complete_prompt = AsyncMock(side_effect=_complete_prompt)
    mock_task_llm.generate = AsyncMock(return_value="Mình có thể giúp bạn viết lại lời chào đó.")

    result = await execute_chat_turn(
        mock_graph,
        mock_task_llm,
        messages=[Message(role="user", content="giúp mình với nhé")],
        max_tokens=None,
        guard=mock_guard,
        prompts=prompts,
    )

    assert result["task_type"] == "general_chat"
    assert "lời chào khách hàng" in result["resolved_query"]
    mock_graph.ainvoke.assert_not_called()


@pytest.mark.asyncio
async def test_streaming_relays_upstream_deltas_without_rechunking():
    """Streaming path calls stream_chat() and relays each delta token immediately."""
    tokens = ["GDP", " tăng", " 6.93%", " [S1]"]
    app, _, mock_task_llm, _, _ = _make_app(
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
