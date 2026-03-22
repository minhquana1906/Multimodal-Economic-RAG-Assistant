"""Tests for nested Pydantic config groups."""
import pytest


def test_llm_config_defaults():
    from orchestrator.config import LLMConfig
    cfg = LLMConfig()
    assert cfg.url == "http://localhost:8004"
    assert cfg.model == "Qwen/Qwen3.5-4B"
    assert cfg.temperature == 0.7
    assert cfg.max_tokens == 512
    assert cfg.timeout == 60.0


def test_services_config_defaults():
    from orchestrator.config import ServicesConfig
    cfg = ServicesConfig()
    assert cfg.embedding_url == "http://embedding:8001"
    assert cfg.embedding_model == "Qwen/Qwen3-Embedding-0.6B"
    assert cfg.reranker_url == "http://reranker:8002"
    assert cfg.reranker_timeout == 15.0
    assert cfg.guard_url == "http://guard:8003"
    assert cfg.qdrant_url == "http://qdrant:6333"
    assert cfg.qdrant_collection == "econ_vn_news"


def test_rag_config_defaults():
    from orchestrator.config import RAGConfig
    cfg = RAGConfig()
    assert cfg.retrieval_top_k == 20
    assert cfg.rerank_top_n == 5
    assert cfg.fallback_min_chunks == 3
    assert cfg.fallback_score_threshold == 0.5
    assert cfg.context_limit == 5
    assert cfg.citation_limit == 5


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


def test_observability_config_defaults():
    from orchestrator.config import ObservabilityConfig
    cfg = ObservabilityConfig()
    assert cfg.log_level == "INFO"
    assert cfg.langsmith_api_key is None
    assert cfg.langsmith_project == "multimodal-economic-rag"
    assert cfg.tavily_api_key is None


def test_settings_nested_access():
    from orchestrator.config import Settings
    s = Settings(_env_file=None)
    assert s.llm.url == "http://localhost:8004"
    assert s.services.embedding_url == "http://embedding:8001"
    assert s.rag.retrieval_top_k == 20
    assert s.prompts.no_context_message.startswith("Xin lỗi")
    assert s.observability.langsmith_api_key is None


def test_settings_env_override(monkeypatch):
    monkeypatch.setenv("RAG__RETRIEVAL_TOP_K", "30")
    monkeypatch.setenv("LLM__TEMPERATURE", "0.3")
    from orchestrator.config import Settings
    s = Settings(_env_file=None)
    assert s.rag.retrieval_top_k == 30
    assert s.llm.temperature == 0.3


def test_get_settings_returns_same_instance():
    from orchestrator.config import get_settings
    s1 = get_settings()
    s2 = get_settings()
    assert s1 is s2


def test_observability_api_keys_from_env(monkeypatch):
    monkeypatch.setenv("OBSERVABILITY__TAVILY_API_KEY", "tvly-test-key")
    monkeypatch.setenv("OBSERVABILITY__LANGSMITH_API_KEY", "ls-test-key")
    from orchestrator.config import Settings
    s = Settings(_env_file=None)
    assert s.observability.tavily_api_key == "tvly-test-key"
    assert s.observability.langsmith_api_key == "ls-test-key"


def test_asr_config_defaults():
    """ServicesConfig has ASR-related fields with correct defaults."""
    from orchestrator.config import ServicesConfig
    cfg = ServicesConfig()
    assert cfg.asr_url == ""
    assert cfg.asr_timeout == 30.0
    assert cfg.asr_max_duration_s == 60


def test_asr_config_from_env(monkeypatch):
    """ASR config can be populated from environment variables."""
    monkeypatch.setenv("SERVICES__ASR_URL", "http://asr:8005")
    monkeypatch.setenv("SERVICES__ASR_TIMEOUT", "15.0")
    monkeypatch.setenv("SERVICES__ASR_MAX_DURATION_S", "120")
    from orchestrator.config import Settings
    s = Settings()
    assert s.services.asr_url == "http://asr:8005"
    assert s.services.asr_timeout == 15.0
    assert s.services.asr_max_duration_s == 120
