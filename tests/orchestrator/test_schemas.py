def test_schemas_validate():
    from orchestrator.models.schemas import ChatRequest, Message
    req = ChatRequest(
        model="test-model",
        messages=[Message(role="user", content="test query")]
    )
    assert len(req.messages) == 1
    assert req.messages[0].role == "user"
    assert req.temperature == 0.7
    assert req.stream == False

def test_chunk_context_defaults():
    from orchestrator.models.schemas import ChunkContext
    chunk = ChunkContext(text="some text")
    assert chunk.source == ""
    assert chunk.score == 0.0
