import pytest


def test_services_config_uses_single_inference_url():
    from orchestrator.config import ServicesConfig

    assert set(ServicesConfig.model_fields.keys()) == {
        "inference_url",
        "inference_timeout",
        "qdrant_url",
        "qdrant_collection",
    }

    services = ServicesConfig(
        inference_url="http://inference:8001",
        inference_timeout=30.0,
        qdrant_url="http://qdrant:6333",
        qdrant_collection="econ_vn_news",
    )
    assert services.inference_url.endswith(":8001")


def test_rag_config_keeps_web_fallback_thresholds():
    from orchestrator.config import RAGConfig

    # Keep the explicit web fallback knobs as first-class config.
    assert "web_fallback_min_chunks" in RAGConfig.model_fields
    assert "web_fallback_hard_threshold" in RAGConfig.model_fields
    assert "web_fallback_soft_threshold" in RAGConfig.model_fields

    config = RAGConfig(
        retrieval_top_k=5,
        rerank_top_n=3,
        web_fallback_min_chunks=2,
        web_fallback_hard_threshold=0.70,
        web_fallback_soft_threshold=0.85,
        context_limit=8,
        citation_limit=6,
    )
    assert config.web_fallback_min_chunks == 2
    assert config.web_fallback_hard_threshold == 0.70
    assert config.web_fallback_soft_threshold == 0.85


def test_schemas_validate():
    from orchestrator.models.schemas import ChatRequest, Message
    req = ChatRequest(
        model="test-model",
        messages=[Message(role="user", content="test query")]
    )
    assert len(req.messages) == 1
    assert req.messages[0].role == "user"
    assert req.temperature == 0.7
    assert req.stream is False

    assert "response_mode" not in ChatRequest.model_fields
    assert "modalities" not in ChatRequest.model_fields
    assert "audio" not in ChatRequest.model_fields


def test_chunk_context_defaults():
    from orchestrator.models.schemas import ChunkContext
    chunk = ChunkContext(text="some text")
    assert chunk.source == ""
    assert chunk.url == ""
    assert chunk.score == 0.0


def test_message_role_validation():
    from orchestrator.models.schemas import Message
    from pydantic import ValidationError
    Message(role="user", content="hi")
    Message(role="system", content="hi")
    Message(role="assistant", content="hi")
    with pytest.raises(ValidationError):
        Message(role="hacker", content="hi")


def test_chat_request_temperature_bounds():
    from orchestrator.models.schemas import ChatRequest, Message
    from pydantic import ValidationError
    msgs = [Message(role="user", content="hi")]
    ChatRequest(model="m", messages=msgs, temperature=0.0)
    ChatRequest(model="m", messages=msgs, temperature=2.0)
    with pytest.raises(ValidationError):
        ChatRequest(model="m", messages=msgs, temperature=3.0)


def test_chat_request_empty_messages_invalid():
    from orchestrator.models.schemas import ChatRequest
    from pydantic import ValidationError
    with pytest.raises(ValidationError):
        ChatRequest(model="m", messages=[])


def test_chat_response_has_openai_fields():
    from orchestrator.models.schemas import (
        AssistantMessage,
        ChatCompletionChoice,
        ChatResponse,
    )
    resp = ChatResponse(
        model="test",
        choices=[ChatCompletionChoice(message=AssistantMessage(content="hello"))]
    )
    assert resp.id.startswith("chatcmpl-")
    assert resp.object == "chat.completion"
    assert resp.created > 0


def test_message_supports_text_content_parts():
    from orchestrator.models.schemas import Message, TextContentPart

    message = Message(
        role="user",
        content=[TextContentPart(text="GDP"), TextContentPart(text=" Việt Nam")],
    )

    assert message.text_content() == "GDP Việt Nam"


def test_stream_chunk_uses_delta_contract():
    from orchestrator.models.schemas import ChatDelta, ChatStreamChoice, ChatStreamChunk

    chunk = ChatStreamChunk(
        model="test",
        choices=[ChatStreamChoice(delta=ChatDelta(role="assistant", content="hi"))],
    )

    assert chunk.choices[0].delta.content == "hi"


def test_image_content_part_round_trips():
    from orchestrator.models.schemas import ImageContentPart, ImageUrl
    part = ImageContentPart(image_url=ImageUrl(url="data:image/png;base64,abc123"))
    assert part.type == "image_url"
    assert part.image_url.url == "data:image/png;base64,abc123"


def test_message_accepts_image_content_part():
    from orchestrator.models.schemas import Message, ImageContentPart, ImageUrl, TextContentPart
    msg = Message(
        role="user",
        content=[
            TextContentPart(text="What is this?"),
            ImageContentPart(image_url=ImageUrl(url="data:image/png;base64,abc")),
        ],
    )
    assert msg.text_content() == "What is this?"


def test_message_text_content_ignores_image_parts():
    from orchestrator.models.schemas import Message, ImageContentPart, ImageUrl
    msg = Message(
        role="user",
        content=[ImageContentPart(image_url=ImageUrl(url="data:image/png;base64,xyz"))],
    )
    assert msg.text_content() == ""
