import logging
import os

import pytest

from orchestrator.config import ObservabilityConfig


def test_setup_logging_registers_domain_levels():
    from orchestrator.tracing import setup_logging, DOMAIN_LEVELS
    setup_logging(ObservabilityConfig())
    from loguru import logger
    for name, no, _ in DOMAIN_LEVELS:
        lvl = logger.level(name)
        assert lvl.no == no


def test_setup_logging_intercepts_stdlib():
    from orchestrator.tracing import setup_logging
    setup_logging(ObservabilityConfig())
    root = logging.getLogger()
    assert any(h.__class__.__name__ == "_InterceptHandler" for h in root.handlers)


def test_setup_langsmith_no_key_does_not_set_env(monkeypatch):
    monkeypatch.delenv("LANGCHAIN_TRACING_V2", raising=False)
    from orchestrator.tracing import setup_langsmith
    setup_langsmith(ObservabilityConfig(langsmith_api_key=None))
    assert "LANGCHAIN_TRACING_V2" not in os.environ


def test_setup_langsmith_with_key_sets_env(monkeypatch):
    from orchestrator.tracing import setup_langsmith
    setup_langsmith(
        ObservabilityConfig(langsmith_api_key="ls-test", langsmith_project="proj")
    )
    assert os.environ["LANGCHAIN_TRACING_V2"] == "true"
    assert os.environ["LANGCHAIN_API_KEY"] == "ls-test"
    assert os.environ["LANGCHAIN_PROJECT"] == "proj"
