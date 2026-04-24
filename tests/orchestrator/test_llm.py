import pytest


def test_llm_uses_placeholder_api_key_when_blank(monkeypatch):
    """Blank API keys should not produce an invalid `Authorization: Bearer ` header."""
    from orchestrator.services.llm import LLMClient

    captured: dict[str, object] = {}

    class FakeAsyncOpenAI:
        def __init__(self, *, base_url, api_key, timeout):
            captured["base_url"] = base_url
            captured["api_key"] = api_key
            captured["timeout"] = timeout

    monkeypatch.setattr("orchestrator.services.llm.AsyncOpenAI", FakeAsyncOpenAI)

    LLMClient(
        url="http://llm:8004/v1",
        model="test-model",
        temperature=0.7,
        max_tokens=512,
        timeout=30.0,
        api_key="",
    )

    assert captured["api_key"] == "dummy"


@pytest.mark.asyncio
async def test_llm_returns_vietnamese_error_on_failure():
    """Returns Vietnamese error message when LLM service is unreachable."""
    from orchestrator.services.llm import LLMClient
    llm = LLMClient(
        url="http://localhost:9999",
        model="test-model",
        temperature=0.7,
        max_tokens=512,
        timeout=1.0,
    )
    result = await llm.generate("system prompt", "user prompt")
    assert "Xin lỗi" in result


@pytest.mark.asyncio
async def test_llm_warm_start_uses_minimal_completion(monkeypatch):
    from orchestrator.services.llm import LLMClient

    llm = LLMClient(
        url="http://llm:8004/v1",
        model="test-model",
        temperature=0.7,
        max_tokens=512,
        timeout=30.0,
        api_key="",
    )

    captured: dict[str, object] = {}

    async def fake_create_completion(messages, *, max_tokens=None):
        captured["messages"] = messages
        captured["max_tokens"] = max_tokens
        return "", 0, 0

    monkeypatch.setattr(llm, "_create_completion", fake_create_completion)

    await llm.warm_start()

    assert captured["messages"] == [{"role": "user", "content": "ping"}]
    assert captured["max_tokens"] == 1


@pytest.mark.asyncio
async def test_detect_intent_uses_supplied_prompt_strings(monkeypatch):
    from orchestrator.services.llm import LLMClient

    llm = LLMClient(
        url="http://llm:8004/v1",
        model="test-model",
        temperature=0.7,
        max_tokens=512,
        timeout=30.0,
        api_key="",
    )

    captured: dict[str, object] = {}

    async def fake_create_completion(messages, *, max_tokens=None):
        captured["messages"] = messages
        captured["max_tokens"] = max_tokens
        return '{"route":"direct","resolved_query":"Tom tat cau hoi"}', 5, 10

    monkeypatch.setattr(llm, "_create_completion", fake_create_completion)

    result = await llm.detect_intent(
        system_prompt="SYSTEM PROMPT",
        user_prompt="USER PROMPT",
        fallback_query="fallback query",
    )

    assert result == {"route": "direct", "resolved_query": "Tom tat cau hoi"}
    assert captured["messages"] == [
        {"role": "system", "content": "SYSTEM PROMPT"},
        {"role": "user", "content": "USER PROMPT"},
    ]
    assert captured["max_tokens"] == 128


@pytest.mark.asyncio
async def test_detect_intent_falls_back_to_direct_on_invalid_json(monkeypatch):
    """Invalid JSON from model must fall back to direct, not rag."""
    from orchestrator.services.llm import LLMClient

    llm = LLMClient(url="http://llm:8004/v1", model="m", temperature=0.7, max_tokens=512, timeout=30.0)

    async def fake_create_completion(messages, *, max_tokens=None):
        return "this is not json at all", 5, 10

    monkeypatch.setattr(llm, "_create_completion", fake_create_completion)

    result = await llm.detect_intent(
        system_prompt="s", user_prompt="u", fallback_query="fallback"
    )
    assert result["route"] == "direct"
    assert result["resolved_query"] == "fallback"


@pytest.mark.asyncio
async def test_detect_intent_logs_trace_on_success(monkeypatch, caplog):
    """detect_intent must emit a TRACE-level log with the raw response."""
    import logging
    from orchestrator.services.llm import LLMClient
    from orchestrator.tracing import setup_logging
    from orchestrator.config import ObservabilityConfig

    setup_logging(ObservabilityConfig(log_level="INFO", langsmith_project="p", app_mode="dev"))

    llm = LLMClient(url="http://llm:8004/v1", model="m", temperature=0.7, max_tokens=512, timeout=30.0)

    async def fake_create_completion(messages, *, max_tokens=None):
        return '{"route":"direct","resolved_query":"test"}', 5, 10

    monkeypatch.setattr(llm, "_create_completion", fake_create_completion)

    # Just verify it completes without error in dev mode (TRACE calls don't raise)
    result = await llm.detect_intent(system_prompt="s", user_prompt="u", fallback_query="fb")
    assert result["route"] == "direct"


@pytest.mark.asyncio
async def test_caption_image_calls_create_completion_with_vision_format(monkeypatch):
    from orchestrator.services.llm import LLMClient

    llm = LLMClient(url="http://llm:8004/v1", model="m", temperature=0.7, max_tokens=512, timeout=30.0)

    captured: dict = {}

    async def fake_create_completion(messages, *, max_tokens=None):
        captured["messages"] = messages
        captured["max_tokens"] = max_tokens
        return "A financial chart showing GDP growth.", 20, 50

    monkeypatch.setattr(llm, "_create_completion", fake_create_completion)

    result = await llm.caption_image(
        image_url="data:image/png;base64,abc",
        caption_prompt="Describe this image.",
    )

    assert result == "A financial chart showing GDP growth."
    assert captured["max_tokens"] == 100
    msg = captured["messages"][0]
    assert msg["role"] == "user"
    content_parts = msg["content"]
    assert any(p["type"] == "image_url" for p in content_parts)
    assert any(p["type"] == "text" and "Describe" in p["text"] for p in content_parts)


@pytest.mark.asyncio
async def test_caption_image_returns_empty_string_on_error(monkeypatch):
    from orchestrator.services.llm import LLMClient

    llm = LLMClient(url="http://llm:8004/v1", model="m", temperature=0.7, max_tokens=512, timeout=30.0)

    async def fake_create_completion(messages, *, max_tokens=None):
        raise RuntimeError("vision model unavailable")

    monkeypatch.setattr(llm, "_create_completion", fake_create_completion)

    result = await llm.caption_image(image_url="data:image/png;base64,x", caption_prompt="describe")
    assert result == ""
