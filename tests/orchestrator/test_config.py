def test_config_loads_defaults():
    from orchestrator.config import Settings
    s = Settings()
    assert s.embedding_url == "http://embedding:8001"
    assert s.retrieval_top_k == 20
    assert s.langsmith_project == "multimodal-rag"
    assert s.fallback_min_chunks == 3


def test_config_timeout_defaults():
    from orchestrator.config import Settings
    s = Settings()
    assert s.guard_timeout == 10.0
    assert s.llm_timeout == 60.0
    assert s.embedding_timeout == 15.0
    assert s.reranker_timeout == 15.0


def test_config_langsmith_key_is_none_by_default(monkeypatch):
    monkeypatch.delenv("LANGSMITH_API_KEY", raising=False)
    from orchestrator.config import Settings
    s = Settings(_env_file=None)
    assert s.langsmith_api_key is None


def test_get_settings_returns_same_instance():
    from orchestrator.config import get_settings
    s1 = get_settings()
    s2 = get_settings()
    assert s1 is s2
