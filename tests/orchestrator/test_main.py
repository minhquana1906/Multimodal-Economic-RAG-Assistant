from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock

import pytest
from fastapi import APIRouter


@pytest.mark.asyncio
async def test_create_app_wires_inference_client(monkeypatch):
    """App startup should wire InferenceClient into the shared services namespace."""
    import orchestrator.main as main

    captured: dict[str, object] = {}
    inference_instance = object()

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
        inference_url="http://inference:8001",
        inference_timeout=30.0,
        qdrant_url="http://qdrant",
        qdrant_collection="econ_vn_news",
    )
    settings.prompts = MagicMock()

    monkeypatch.setattr(main, "get_settings", lambda: settings)
    monkeypatch.setattr(main, "setup_logging", lambda config: None)
    monkeypatch.setattr(main, "setup_langsmith", lambda config: None)
    monkeypatch.setattr(main, "InferenceClient", lambda *args, **kwargs: inference_instance)
    monkeypatch.setattr(main, "RetrieverClient", lambda *args, **kwargs: object())
    llm_client = SimpleNamespace(warm_start=AsyncMock())
    monkeypatch.setattr(main, "LLMClient", lambda *args, **kwargs: llm_client)
    monkeypatch.setattr(main, "WebSearchClient", lambda *args, **kwargs: object())

    def fake_build_rag_graph(services, config, **kwargs):
        captured["services"] = services
        return object()

    monkeypatch.setattr(main, "build_rag_graph", fake_build_rag_graph)
    monkeypatch.setattr(
        main,
        "create_chat_router",
        lambda *args, **kwargs: APIRouter(),
    )

    app = main.create_app()
    async with app.router.lifespan_context(app):
        pass

    services = captured["services"]
    assert services.inference is inference_instance
    llm_client.warm_start.assert_awaited_once()


@pytest.mark.asyncio
async def test_create_app_startup_ignores_warm_start_failure(monkeypatch):
    import orchestrator.main as main

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
        inference_url="http://inference:8001",
        inference_timeout=30.0,
        qdrant_url="http://qdrant",
        qdrant_collection="econ_vn_news",
    )
    settings.prompts = MagicMock()

    monkeypatch.setattr(main, "get_settings", lambda: settings)
    monkeypatch.setattr(main, "setup_logging", lambda config: None)
    monkeypatch.setattr(main, "setup_langsmith", lambda config: None)
    monkeypatch.setattr(main, "InferenceClient", lambda *args, **kwargs: object())
    monkeypatch.setattr(main, "RetrieverClient", lambda *args, **kwargs: object())
    monkeypatch.setattr(main, "WebSearchClient", lambda *args, **kwargs: object())
    monkeypatch.setattr(main, "build_rag_graph", lambda *args, **kwargs: object())
    monkeypatch.setattr(main, "create_chat_router", lambda *args, **kwargs: APIRouter())

    llm_client = SimpleNamespace(warm_start=AsyncMock(side_effect=RuntimeError("cold start")))
    monkeypatch.setattr(main, "LLMClient", lambda *args, **kwargs: llm_client)

    app = main.create_app()
    async with app.router.lifespan_context(app):
        pass

    llm_client.warm_start.assert_awaited_once()
