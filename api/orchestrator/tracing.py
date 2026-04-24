from __future__ import annotations

import logging
import os
import sys
import time
from contextlib import asynccontextmanager
from collections.abc import AsyncIterator

from loguru import logger

from orchestrator.config import ObservabilityConfig

DOMAIN_LEVELS = [
    ("RETRIEVAL", 25, "<cyan>"),
    ("RERANK",    26, "<blue>"),
    ("LLM",       28, "<magenta>"),
]

LOG_FORMAT = (
    "{time:YYYY-MM-DD HH:mm:ss.SSS} | {level:<10} | "
    "{name}:{function}:{line} | {extra[request_id]} | {message}"
)


class _InterceptHandler(logging.Handler):
    """Route stdlib log records through loguru."""

    def emit(self, record: logging.LogRecord) -> None:
        try:
            level = logger.level(record.levelname).name
        except ValueError:
            level = record.levelno
        frame, depth = sys._getframe(6), 6
        while frame.f_code.co_filename == logging.__file__:
            frame = frame.f_back
            depth += 1
        logger.opt(depth=depth, exception=record.exc_info).log(
            level, record.getMessage()
        )


def setup_logging(config: ObservabilityConfig) -> None:
    """Configure loguru: remove default sink, add structured stderr sink,
    register custom domain levels, intercept stdlib logging."""
    logger.remove()

    logger.configure(extra={"request_id": "-"})

    effective_level = "TRACE" if config.app_mode == "dev" else config.log_level
    logger.add(sys.stderr, format=LOG_FORMAT, level=effective_level, colorize=True)

    for name, no, color in DOMAIN_LEVELS:
        try:
            logger.level(name, no=no, color=color)
        except (TypeError, ValueError):
            pass  

    logging.basicConfig(handlers=[_InterceptHandler()], level=0, force=True)


@asynccontextmanager
async def log_phase(phase: str, **meta) -> AsyncIterator[dict]:
    """Async context manager that logs phase name and elapsed_ms on exit.

    Yields a mutable dict so callers can populate fields before the log fires:
        async with log_phase("retrieval", top_k=10) as ctx:
            docs = await search(...)
            ctx["hits"] = len(docs)
    """
    t0 = time.monotonic()
    ctx: dict = dict(meta)
    yield ctx
    elapsed_ms = int((time.monotonic() - t0) * 1000)
    parts = [f"phase={phase}", f"ms={elapsed_ms}"]
    parts.extend(f"{k}={v}" for k, v in ctx.items())
    logger.info(" ".join(parts))


def setup_langsmith(config: ObservabilityConfig) -> None:
    """Enable LangSmith tracing if api_key is set."""
    if not config.langsmith_api_key:
        logger.info("LangSmith API key not set; tracing disabled")
        return
    os.environ["LANGCHAIN_TRACING_V2"] = "true"
    os.environ["LANGCHAIN_API_KEY"] = config.langsmith_api_key
    os.environ["LANGCHAIN_PROJECT"] = config.langsmith_project
    logger.info(f"LangSmith tracing enabled for project: {config.langsmith_project}")
