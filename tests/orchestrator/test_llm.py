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
