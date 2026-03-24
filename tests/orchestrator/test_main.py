from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest
from fastapi import APIRouter


@pytest.mark.asyncio
async def test_create_app_wires_sparse_encoder_service(monkeypatch):
    """App startup should register SparseEncoderService in the shared services namespace."""
    import orchestrator.main as main

    captured: dict[str, object] = {}
    sparse_encoder_instance = object()

    settings = MagicMock()
    settings.observability = MagicMock()
    settings.observability.tavily_api_key = ""
    settings.llm = MagicMock()
    settings.llm.url = "http://llm"
    settings.llm.model = "test-model"
    settings.llm.temperature = 0.0
    settings.llm.max_tokens = 128
    settings.llm.timeout = 30.0
    settings.llm.api_key = ""
    settings.services = SimpleNamespace(
        guard_url="http://guard",
        guard_timeout=5.0,
        embedding_url="http://embed",
        embedding_timeout=5.0,
        qdrant_url="http://qdrant",
        qdrant_collection="econ_vn_news",
        reranker_url="http://reranker",
        reranker_timeout=5.0,
        asr_url="http://asr",
        asr_timeout=15.0,
        tts_url="http://tts",
        tts_timeout=25.0,
    )

    monkeypatch.setattr(main, "get_settings", lambda: settings)
    monkeypatch.setattr(main, "setup_logging", lambda config: None)
    monkeypatch.setattr(main, "setup_langsmith", lambda config: None)
    monkeypatch.setattr(main, "GuardClient", lambda *args, **kwargs: object())
    monkeypatch.setattr(main, "EmbedderClient", lambda *args, **kwargs: object())
    monkeypatch.setattr(main, "RetrieverClient", lambda *args, **kwargs: object())
    monkeypatch.setattr(main, "RerankerClient", lambda *args, **kwargs: object())
    monkeypatch.setattr(main, "LLMClient", lambda *args, **kwargs: object())
    monkeypatch.setattr(main, "WebSearchClient", lambda *args, **kwargs: object())
    monkeypatch.setattr(main, "ASRClient", lambda *args, **kwargs: object())
    monkeypatch.setattr(main, "TTSClient", lambda *args, **kwargs: object())
    monkeypatch.setattr(
        main,
        "SparseEncoderService",
        lambda *args, **kwargs: sparse_encoder_instance,
        raising=False,
    )

    def fake_build_rag_graph(services, config):
        captured["services"] = services
        captured["config"] = config
        return object()

    monkeypatch.setattr(main, "build_rag_graph", fake_build_rag_graph)
    monkeypatch.setattr(main, "create_chat_router", lambda graph, llm: APIRouter())
    monkeypatch.setattr(main, "create_audio_router", lambda asr, tts: APIRouter())

    app = main.create_app()
    async with app.router.lifespan_context(app):
        pass

    services = captured["services"]
    assert services.sparse_encoder is sparse_encoder_instance
    assert services.asr is not None
    assert services.tts is not None
