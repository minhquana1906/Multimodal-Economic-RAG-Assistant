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
    assert req.stream is False


def test_chunk_context_defaults():
    from orchestrator.models.schemas import ChunkContext
    chunk = ChunkContext(text="some text")
    assert chunk.source == ""
    assert chunk.score == 0.0


def test_message_role_validation():
    from orchestrator.models.schemas import Message
    from pydantic import ValidationError
    # Valid roles
    Message(role="user", content="hi")
    Message(role="system", content="hi")
    Message(role="assistant", content="hi")
    # Invalid role raises
    with pytest.raises(ValidationError):
        Message(role="hacker", content="hi")


def test_chat_request_temperature_bounds():
    from orchestrator.models.schemas import ChatRequest, Message
    from pydantic import ValidationError
    msgs = [Message(role="user", content="hi")]
    # Valid
    ChatRequest(model="m", messages=msgs, temperature=0.0)
    ChatRequest(model="m", messages=msgs, temperature=2.0)
    # Invalid
    with pytest.raises(ValidationError):
        ChatRequest(model="m", messages=msgs, temperature=3.0)


def test_chat_request_empty_messages_invalid():
    from orchestrator.models.schemas import ChatRequest
    from pydantic import ValidationError
    with pytest.raises(ValidationError):
        ChatRequest(model="m", messages=[])


def test_chat_response_has_openai_fields():
    from orchestrator.models.schemas import ChatResponse, ChatChoice, ChatDelta
    resp = ChatResponse(
        model="test",
        choices=[ChatChoice(delta=ChatDelta(content="hello"))]
    )
    assert resp.id.startswith("chatcmpl-")
    assert resp.object == "chat.completion"
    assert resp.created > 0
