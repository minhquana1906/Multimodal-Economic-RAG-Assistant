from __future__ import annotations

import logging

from fastapi import FastAPI

from orchestrator.config import get_settings
from orchestrator.tracing import setup_langsmith
from orchestrator.pipeline.rag import build_rag_graph
from orchestrator.routers.chat import create_chat_router
from orchestrator.services.guard import GuardClient
from orchestrator.services.embedder import EmbedderClient
from orchestrator.services.retriever import RetrieverClient
from orchestrator.services.reranker import RerankerClient
from orchestrator.services.llm import LLMClient
from orchestrator.services.web_search import WebSearchClient

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# ── App factory ──────────────────────────────────────────────────────────────

def create_app() -> FastAPI:
    """Create and configure the FastAPI application."""
    settings = get_settings()
    setup_langsmith(settings)

    app = FastAPI(
        title="Multimodal RAG Orchestrator",
        description="OpenAI-compatible RAG API for Vietnamese economic news",
        version="1.0.0",
    )

    @app.get("/health")
    async def health():
        return {"status": "ok", "model": "Qwen/Qwen3.5-4B"}

    @app.get("/v1/models")
    async def models():
        return {"data": [{"id": "multimodal-rag", "object": "model"}]}

    @app.on_event("startup")
    async def startup() -> None:
        logger.info("Initialising service clients and RAG graph…")

        class Services:
            guard = GuardClient(settings.guard_url, settings.guard_timeout)
            embedder = EmbedderClient(settings.embedding_url, settings.embedding_timeout)
            retriever = RetrieverClient(settings.qdrant_url, settings.qdrant_collection)
            reranker = RerankerClient(settings.reranker_url, settings.reranker_timeout)
            llm = LLMClient(settings.llm_url, timeout=settings.llm_timeout)
            web_search = WebSearchClient(api_key=settings.langsmith_api_key or "")

        rag_graph = build_rag_graph(Services(), settings)
        chat_router = create_chat_router(rag_graph)
        app.include_router(chat_router)

        logger.info("RAG graph ready")

    return app


app = create_app()
