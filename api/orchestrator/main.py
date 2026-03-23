from __future__ import annotations

from contextlib import asynccontextmanager
from types import SimpleNamespace

from fastapi import FastAPI
from loguru import logger

from orchestrator.config import get_settings
from orchestrator.tracing import setup_logging, setup_langsmith
from orchestrator.pipeline.rag import build_rag_graph
from orchestrator.routers.chat import create_chat_router
from orchestrator.services.guard import GuardClient
from orchestrator.services.embedder import EmbedderClient
from orchestrator.services.retriever import RetrieverClient
from orchestrator.services.reranker import RerankerClient
from orchestrator.services.llm import LLMClient
from orchestrator.services.web_search import WebSearchClient


def create_app() -> FastAPI:
    """Create and configure the FastAPI application."""
    settings = get_settings()
    setup_logging(settings.observability)
    setup_langsmith(settings.observability)

    @asynccontextmanager
    async def lifespan(app: FastAPI):
        logger.info("Initialising service clients and RAG graph…")
        services = SimpleNamespace(
            guard=GuardClient(
                settings.services.guard_url,
                settings.services.guard_timeout,
            ),
            embedder=EmbedderClient(
                settings.services.embedding_url,
                settings.services.embedding_timeout,
            ),
            retriever=RetrieverClient(
                settings.services.qdrant_url,
                settings.services.qdrant_collection,
            ),
            reranker=RerankerClient(
                settings.services.reranker_url,
                settings.services.reranker_timeout,
            ),
            llm=LLMClient(
                url=settings.llm.url,
                model=settings.llm.model,
                temperature=settings.llm.temperature,
                max_tokens=settings.llm.max_tokens,
                timeout=settings.llm.timeout,
                api_key=settings.llm.api_key or "",
            ),
            web_search=WebSearchClient(
                api_key=settings.observability.tavily_api_key or "",
            ),
        )
        rag_graph = build_rag_graph(services, settings)
        app.include_router(create_chat_router(rag_graph, services.llm))
        logger.info("RAG graph ready")
        yield
        # graceful shutdown placeholder

    app = FastAPI(
        title="Multimodal RAG Orchestrator",
        description="OpenAI-compatible RAG API for Vietnamese economic news",
        version="1.0.0",
        lifespan=lifespan,
    )

    @app.get("/health")
    async def health():
        return {"status": "ok", "model": settings.llm.model}

    @app.get("/v1/models")
    async def models():
        return {"data": [{"id": "multimodal-economic-rag", "object": "model"}]}

    return app


app = create_app()
