def test_config_loads_defaults():
    from orchestrator.config import Settings
    s = Settings()
    assert s.embedding_url == "http://embedding:8001"
    assert s.retrieval_top_k == 20
    assert s.langsmith_project == "multimodal-rag"
    assert s.fallback_min_chunks == 3
