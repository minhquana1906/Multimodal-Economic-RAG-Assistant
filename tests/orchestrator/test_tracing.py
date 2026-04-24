import logging
import os
from unittest.mock import AsyncMock, MagicMock

import pytest

from orchestrator.config import ObservabilityConfig


def _observability_config(**overrides):
    payload = {
        "log_level": "INFO",
        "langsmith_project": "multimodal-economic-rag",
        "langsmith_api_key": None,
        "tavily_api_key": None,
    }
    payload.update(overrides)
    return ObservabilityConfig(**payload)


def test_setup_logging_registers_domain_levels():
    from orchestrator.tracing import setup_logging, DOMAIN_LEVELS
    setup_logging(_observability_config())
    from loguru import logger
    for name, no, _ in DOMAIN_LEVELS:
        lvl = logger.level(name)
        assert lvl.no == no


def test_setup_logging_intercepts_stdlib():
    from orchestrator.tracing import setup_logging
    setup_logging(_observability_config())
    root = logging.getLogger()
    assert any(h.__class__.__name__ == "_InterceptHandler" for h in root.handlers)


def test_setup_langsmith_no_key_does_not_set_env(monkeypatch):
    monkeypatch.delenv("LANGCHAIN_TRACING_V2", raising=False)
    from orchestrator.tracing import setup_langsmith
    setup_langsmith(_observability_config(langsmith_api_key=None))
    assert "LANGCHAIN_TRACING_V2" not in os.environ


def test_setup_langsmith_with_key_sets_env(monkeypatch):
    from orchestrator.tracing import setup_langsmith
    setup_langsmith(_observability_config(langsmith_api_key="ls-test", langsmith_project="proj"))
    assert os.environ["LANGCHAIN_TRACING_V2"] == "true"
    assert os.environ["LANGCHAIN_API_KEY"] == "ls-test"
    assert os.environ["LANGCHAIN_PROJECT"] == "proj"


def test_setup_logging_uses_trace_level_in_dev_mode():
    import sys
    from loguru import logger
    from orchestrator.config import ObservabilityConfig
    from orchestrator.tracing import setup_logging

    cfg = ObservabilityConfig(log_level="INFO", langsmith_project="proj", app_mode="dev")
    setup_logging(cfg)

    # Find the stderr sink and verify its effective level is TRACE (no=5)
    handlers = logger._core.handlers  # loguru internal
    levels = [h._levelno for h in handlers.values()]
    assert any(lvl <= 5 for lvl in levels), f"Expected TRACE sink, got levels: {levels}"


@pytest.mark.asyncio
async def test_execute_chat_turn_root_output_excludes_generation_prompt():
    from orchestrator.routers.chat import execute_chat_turn

    mock_graph = MagicMock()
    mock_graph.ainvoke = AsyncMock(
        return_value={
            "answer": "GDP tăng 7%",
            "citations": [{"context_id": "hybrid:1"}],
            "generation_prompt": "Context that should not appear",
            "final_context": [{"context_id": "hybrid:1"}],
        }
    )

    result = await execute_chat_turn(
        mock_graph,
        None,
        messages=[{"role": "user", "content": "GDP Việt Nam?"}],
        max_tokens=None,
    )

    assert result["answer"] == "GDP tăng 7%"
    assert result["citations"] == [{"context_id": "hybrid:1"}]
    assert result["task_type"] == "rag"
    assert result["resolved_query"] == "GDP Việt Nam?"
    assert "generation_prompt" not in result
    assert "final_context" not in result
