"""Tests for env-driven orchestrator settings."""

from pathlib import Path

import pytest
from pydantic import ValidationError


def _write_env_file(tmp_path: Path) -> Path:
    env_file = tmp_path / "test.env"
    env_file.write_text(
        "\n".join(
            [
                "LLM__URL=http://llm:8004/v1",
                "LLM__MODEL=Qwen/Qwen3.5-4B",
                "LLM__TEMPERATURE=0.3",
                "LLM__MAX_TOKENS=1024",
                "LLM__TIMEOUT=45.0",
                "SERVICES__INFERENCE_URL=http://inference:8001",
                "SERVICES__INFERENCE_TIMEOUT=30.0",
                "SERVICES__QDRANT_URL=http://qdrant:6333",
                "SERVICES__QDRANT_COLLECTION=econ_vn_news",
                "RAG__RETRIEVAL_TOP_K=20",
                "RAG__RERANK_TOP_N=5",
                "RAG__WEB_FALLBACK_MIN_CHUNKS=2",
                "RAG__WEB_FALLBACK_HARD_THRESHOLD=0.7",
                "RAG__WEB_FALLBACK_SOFT_THRESHOLD=0.85",
                "RAG__CONTEXT_LIMIT=5",
                "RAG__CITATION_LIMIT=5",
                "OBSERVABILITY__LOG_LEVEL=INFO",
                "OBSERVABILITY__LANGSMITH_PROJECT=multimodal-economic-rag",
                "OBSERVABILITY__TAVILY_API_KEY=tvly-test-key",
                "",
            ]
        ),
        encoding="utf-8",
    )
    return env_file


def test_llm_config_requires_runtime_values():
    from orchestrator.config import LLMConfig

    with pytest.raises(ValidationError):
        LLMConfig()


def test_services_config_requires_runtime_values():
    from orchestrator.config import ServicesConfig

    with pytest.raises(ValidationError):
        ServicesConfig()


def test_settings_load_nested_runtime_values_from_env_file(tmp_path):
    from orchestrator.config import Settings

    env_file = _write_env_file(tmp_path)
    settings = Settings(_env_file=env_file)

    assert settings.llm.url == "http://llm:8004/v1"
    assert settings.llm.model == "Qwen/Qwen3.5-4B"
    assert settings.llm.temperature == 0.3
    assert settings.services.inference_url == "http://inference:8001"
    assert settings.services.inference_timeout == 30.0
    assert settings.services.qdrant_collection == "econ_vn_news"
    assert settings.rag.retrieval_top_k == 20
    assert settings.observability.log_level == "INFO"
    assert settings.observability.tavily_api_key == "tvly-test-key"


def test_settings_accepts_legacy_rag_env_names(tmp_path):
    from orchestrator.config import Settings

    env_file = tmp_path / "legacy.env"
    env_file.write_text(
        "\n".join(
            [
                "LLM__URL=http://llm:8004/v1",
                "LLM__MODEL=Qwen/Qwen3.5-4B",
                "LLM__TEMPERATURE=0.3",
                "LLM__MAX_TOKENS=1024",
                "LLM__TIMEOUT=45.0",
                "SERVICES__INFERENCE_URL=http://inference:8001",
                "SERVICES__INFERENCE_TIMEOUT=30.0",
                "SERVICES__QDRANT_URL=http://qdrant:6333",
                "SERVICES__QDRANT_COLLECTION=econ_vn_news",
                "RAG__RETRIEVAL_TOP_K=20",
                "RAG__RERANK_TOP_N=5",
                "RAG__FALLBACK_MIN_CHUNKS=3",
                "RAG__FALLBACK_SCORE_THRESHOLD=0.5",
                "RAG__CONTEXT_LIMIT=5",
                "RAG__CITATION_LIMIT=5",
                "OBSERVABILITY__LOG_LEVEL=INFO",
                "OBSERVABILITY__LANGSMITH_PROJECT=multimodal-economic-rag",
                "",
            ]
        ),
        encoding="utf-8",
    )

    settings = Settings(_env_file=env_file)

    assert settings.rag.web_fallback_min_chunks == 3
    assert settings.rag.web_fallback_hard_threshold == 0.5
    assert settings.rag.web_fallback_soft_threshold == 0.85
    assert settings.rag.fallback_min_chunks == 3
    assert settings.rag.fallback_score_threshold == 0.5


def test_settings_env_overrides_env_file(monkeypatch, tmp_path):
    from orchestrator.config import Settings

    env_file = _write_env_file(tmp_path)
    monkeypatch.setenv("RAG__RETRIEVAL_TOP_K", "30")
    monkeypatch.setenv("LLM__TEMPERATURE", "0.7")

    settings = Settings(_env_file=env_file)

    assert settings.rag.retrieval_top_k == 30
    assert settings.llm.temperature == 0.7


def test_prompts_config_defaults():
    from orchestrator.config import PromptsConfig

    cfg = PromptsConfig()
    assert "route" in cfg.intent_system_prompt.lower()
    assert "{messages}" in cfg.intent_user_template
    assert "Viết lại" in cfg.direct_system_prompt
    assert "header `##`" in cfg.direct_response_contract
    assert "{conversation}" in cfg.direct_user_template
    assert "{question}" in cfg.direct_user_template
    assert "kinh tế" in cfg.rag_system_prompt
    assert "{context}" in cfg.rag_user_template
    assert "{question}" in cfg.rag_user_template
    assert cfg.no_context_message.startswith("Không tìm thấy dữ liệu")


def test_get_settings_returns_same_instance(monkeypatch):
    import orchestrator.config as config

    created: list[object] = []

    class DummySettings:
        def __init__(self):
            created.append(self)

    monkeypatch.setattr(config, "Settings", DummySettings)
    config.get_settings.cache_clear()

    first = config.get_settings()
    second = config.get_settings()

    assert first is second
    assert len(created) == 1


def test_observability_api_keys_can_be_omitted(tmp_path):
    from orchestrator.config import Settings

    env_file = _write_env_file(tmp_path)
    settings = Settings(_env_file=env_file)

    assert settings.observability.langsmith_api_key is None
