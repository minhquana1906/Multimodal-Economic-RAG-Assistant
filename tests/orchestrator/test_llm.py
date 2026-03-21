import pytest


@pytest.mark.asyncio
async def test_llm_returns_vietnamese_error_on_failure():
    """Returns Vietnamese error message when LLM service is unreachable."""
    from orchestrator.services.llm import LLMClient
    llm = LLMClient("http://localhost:9999")
    result = await llm.generate("system prompt", "user prompt")
    assert "Xin lỗi" in result
