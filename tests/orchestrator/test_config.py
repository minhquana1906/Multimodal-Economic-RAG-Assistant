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
                "LLM__MODEL=Qwen/Qwen3-4B-Instruct-2507",
                "LLM__TEMPERATURE=0.3",
                "LLM__MAX_TOKENS=1024",
                "LLM__TIMEOUT=45.0",
                "SERVICES__EMBEDDING_URL=http://embedding:8001",
                "SERVICES__EMBEDDING_MODEL=Qwen/Qwen3-Embedding-0.6B",
                "SERVICES__EMBEDDING_TIMEOUT=15.0",
                "SERVICES__EMBEDDING_MAX_SEQ_LENGTH=1024",
                "SERVICES__EMBEDDING_ENCODE_BATCH_SIZE=128",
                "SERVICES__RERANKER_URL=http://reranker:8002",
                "SERVICES__RERANKER_MODEL=Qwen/Qwen3-Reranker-0.6B",
                "SERVICES__RERANKER_TIMEOUT=15.0",
                "SERVICES__GUARD_URL=http://guard:8003",
                "SERVICES__GUARD_MODEL=Qwen/Qwen3Guard-Gen-0.6B",
                "SERVICES__GUARD_TIMEOUT=10.0",
                "SERVICES__GUARD_MAX_NEW_TOKENS=64",
                "SERVICES__QDRANT_URL=http://qdrant:6333",
                "SERVICES__QDRANT_COLLECTION=econ_vn_news",
                "SERVICES__ASR_URL=http://asr:8005",
                "SERVICES__ASR_MODEL=Qwen/Qwen3-ASR-1.7B",
                "SERVICES__ASR_TIMEOUT=30.0",
                "SERVICES__ASR_MAX_DURATION_S=60",
                "SERVICES__ASR_IDLE_TIMEOUT=300",
                "SERVICES__TTS_URL=http://tts:8006",
                "SERVICES__TTS_MODEL=pnnbao-ump/VieNeu-TTS",
                "SERVICES__TTS_TIMEOUT=30.0",
                "SERVICES__TTS_SPEED=1.0",
                "SERVICES__TTS_SAMPLE_RATE=24000",
                "SERVICES__TTS_IDLE_TIMEOUT=300",
                "RAG__RETRIEVAL_TOP_K=20",
                "RAG__RERANK_TOP_N=5",
                "RAG__FALLBACK_MIN_CHUNKS=3",
                "RAG__FALLBACK_SCORE_THRESHOLD=0.5",
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
    assert settings.llm.model == "Qwen/Qwen3-4B-Instruct-2507"
    assert settings.llm.temperature == 0.3
    assert settings.services.embedding_url == "http://embedding:8001"
    assert settings.services.embedding_max_seq_length == 1024
    assert settings.services.embedding_encode_batch_size == 128
    assert settings.services.guard_max_new_tokens == 64
    assert settings.services.qdrant_collection == "econ_vn_news"
    assert settings.services.asr_model == "Qwen/Qwen3-ASR-1.7B"
    assert settings.services.tts_model == "pnnbao-ump/VieNeu-TTS"
    assert settings.services.tts_sample_rate == 24000
    assert settings.rag.retrieval_top_k == 20
    assert settings.observability.log_level == "INFO"
    assert settings.observability.tavily_api_key == "tvly-test-key"


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
    assert "kinh tế" in cfg.system_prompt
    assert "{context}" in cfg.user_template
    assert "{question}" in cfg.user_template
    assert cfg.no_context_message.startswith("Xin lỗi")
    assert cfg.apology_message.startswith("Xin lỗi")
    assert cfg.guard_error_message.startswith("Xin lỗi")
    assert "kinh tế" in cfg.reranker_instruction


def test_get_settings_returns_same_instance():
    from orchestrator.config import get_settings

    get_settings.cache_clear()

    first = get_settings()
    second = get_settings()

    assert first is second


def test_observability_api_keys_can_be_omitted(tmp_path):
    from orchestrator.config import Settings

    env_file = _write_env_file(tmp_path)
    settings = Settings(_env_file=env_file)

    assert settings.observability.langsmith_api_key is None
