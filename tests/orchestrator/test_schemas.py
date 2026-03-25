import pytest


def test_schemas_validate():
    from orchestrator.models.schemas import ChatRequest, Message
    req = ChatRequest(
        model="test-model",
        messages=[Message(role="user", content="test query")]
    )
    assert len(req.messages) == 1
    assert req.messages[0].role == "user"
    assert req.temperature == 0.7
    assert req.response_mode == "text"
    assert req.stream is False


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


def test_chat_request_response_mode_validation():
    from orchestrator.models.schemas import ChatRequest, Message
    from pydantic import ValidationError

    msgs = [Message(role="user", content="hi")]
    req = ChatRequest(model="m", messages=msgs, response_mode="audio")
    assert req.response_mode == "audio"

    with pytest.raises(ValidationError):
        ChatRequest(model="m", messages=msgs, response_mode="voice")


def test_chat_request_accepts_audio_standard_fields():
    from orchestrator.models.schemas import ChatRequest, Message

    msgs = [Message(role="user", content="hi")]
    req = ChatRequest(
        model="m",
        messages=msgs,
        modalities=["text", "audio"],
        audio={"voice": "alloy", "format": "wav"},
    )

    assert req.modalities == ["text", "audio"]
    assert req.audio == {"voice": "alloy", "format": "wav"}


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
