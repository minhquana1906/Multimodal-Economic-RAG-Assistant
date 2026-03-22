import os
import logging

logger = logging.getLogger(__name__)


def setup_langsmith(config) -> None:
    """Configure LangSmith tracing from settings. No-op if api_key is not set."""
    if not config.langsmith_api_key:
        logger.info("LangSmith API key not set; tracing disabled")
        return

    os.environ["LANGSMITH_API_KEY"] = config.langsmith_api_key
    os.environ["LANGSMITH_PROJECT"] = config.langsmith_project
    os.environ["LANGSMITH_TRACING"] = "true"

    logger.info(f"LangSmith tracing enabled for project: {config.langsmith_project}")
