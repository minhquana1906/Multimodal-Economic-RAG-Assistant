"""Tests for LLMClient multimodal methods: describe_image, generate_with_images."""
from __future__ import annotations

import json
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from orchestrator.services.llm import LLMClient


@pytest.fixture
def llm_client():
    return LLMClient(
        url="http://fake-llm:8004/v1",
        model="Qwen/Qwen3.5-4B",
        temperature=0.7,
        max_tokens=1024,
        timeout=30.0,
        api_key="dummy",
    )


def _mock_completion(text: str):
    choice = MagicMock()
    choice.message.content = text
    usage = MagicMock()
    usage.completion_tokens = 10
    resp = MagicMock()
    resp.choices = [choice]
    resp.usage = usage
    return resp


class TestDescribeImage:
    async def test_parses_valid_json(self, llm_client):
        payload = json.dumps({"caption": "Bar chart GDP", "rag_query": "GDP Việt Nam 2023"})
        llm_client._client = MagicMock()
        llm_client._client.chat.completions.create = AsyncMock(return_value=_mock_completion(payload))

        result = await llm_client.describe_image(
            user_text="Phân tích biểu đồ",
            image_content_parts=[{"type": "image_url", "image_url": {"url": "data:image/png;base64,x"}}],
            system_prompt="sys",
            user_template="Yêu cầu: {user_text}. Trả JSON.",
        )
        assert result["caption"] == "Bar chart GDP"
        assert result["rag_query"] == "GDP Việt Nam 2023"

    async def test_parses_markdown_wrapped_json(self, llm_client):
        payload = "```json\n{\"caption\": \"Bảng số liệu\", \"rag_query\": \"thị trường chứng khoán\"}\n```"
        llm_client._client = MagicMock()
        llm_client._client.chat.completions.create = AsyncMock(return_value=_mock_completion(payload))

        result = await llm_client.describe_image(
            user_text="",
            image_content_parts=[],
            system_prompt="",
            user_template="{user_text}",
        )
        assert result["caption"] == "Bảng số liệu"

    async def test_fallback_on_invalid_json(self, llm_client):
        llm_client._client = MagicMock()
        llm_client._client.chat.completions.create = AsyncMock(return_value=_mock_completion("not json"))

        result = await llm_client.describe_image(
            user_text="original query",
            image_content_parts=[],
            system_prompt="",
            user_template="{user_text}",
        )
        assert result["caption"] == ""
        assert result["rag_query"] == "original query"

    async def test_fallback_on_exception(self, llm_client):
        llm_client._client = MagicMock()
        llm_client._client.chat.completions.create = AsyncMock(side_effect=RuntimeError("timeout"))

        result = await llm_client.describe_image(
            user_text="fallback q",
            image_content_parts=[],
            system_prompt="",
            user_template="{user_text}",
        )
        assert result["rag_query"] == "fallback q"


class TestGenerateWithImages:
    async def test_builds_multimodal_content(self, llm_client):
        llm_client._client = MagicMock()
        llm_client._client.chat.completions.create = AsyncMock(return_value=_mock_completion("answer text"))

        image_parts = [{"type": "image_url", "image_url": {"url": "data:image/png;base64,abc"}}]
        result = await llm_client.generate_with_images(
            system_prompt="You are helpful.",
            user_text="Phân tích ảnh",
            image_content_parts=image_parts,
        )
        assert result == "answer text"

        # Verify image was included in the call
        call_kwargs = llm_client._client.chat.completions.create.call_args
        messages = call_kwargs.kwargs.get("messages") or call_kwargs.args[0] if call_kwargs.args else []
        if call_kwargs.kwargs.get("messages"):
            messages = call_kwargs.kwargs["messages"]
        else:
            messages = call_kwargs.args[0] if call_kwargs.args else call_kwargs.kwargs.get("messages", [])
        user_msg = next(m for m in messages if m["role"] == "user")
        content = user_msg["content"]
        assert isinstance(content, list)
        types = [part["type"] for part in content]
        assert "image_url" in types
        assert "text" in types

    async def test_fallback_on_error(self, llm_client):
        llm_client._client = MagicMock()
        llm_client._client.chat.completions.create = AsyncMock(side_effect=Exception("connection error"))

        result = await llm_client.generate_with_images(
            system_prompt="sys",
            user_text="q",
            image_content_parts=[],
        )
        assert "Xin lỗi" in result


class TestDetectIntentStillTextOnly:
    async def test_detect_intent_receives_no_images(self, llm_client):
        """detect_intent must stay text-only regardless of images in conversation."""
        payload = json.dumps({"route": "direct", "resolved_query": "hello"})
        llm_client._client = MagicMock()
        llm_client._client.chat.completions.create = AsyncMock(return_value=_mock_completion(payload))

        await llm_client.detect_intent(
            system_prompt="sys",
            user_prompt="plain text",
            fallback_query="q",
        )
        call_kwargs = llm_client._client.chat.completions.create.call_args
        msgs = call_kwargs.kwargs.get("messages", [])
        for msg in msgs:
            assert isinstance(msg["content"], str), "detect_intent must use text-only content"
