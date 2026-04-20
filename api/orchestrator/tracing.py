from __future__ import annotations

import logging
import os
import sys

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

    logger.add(sys.stderr, format=LOG_FORMAT, level=config.log_level, colorize=True)

    for name, no, color in DOMAIN_LEVELS:
        try:
            logger.level(name, no=no, color=color)
        except (TypeError, ValueError):
            pass  

    logging.basicConfig(handlers=[_InterceptHandler()], level=0, force=True)


def setup_langsmith(config: ObservabilityConfig) -> None:
    """Enable LangSmith tracing if api_key is set."""
    if not config.langsmith_api_key:
        logger.info("LangSmith API key not set; tracing disabled")
        return
    os.environ["LANGCHAIN_TRACING_V2"] = "true"
    os.environ["LANGCHAIN_API_KEY"] = config.langsmith_api_key
    os.environ["LANGCHAIN_PROJECT"] = config.langsmith_project
    logger.info(f"LangSmith tracing enabled for project: {config.langsmith_project}")
