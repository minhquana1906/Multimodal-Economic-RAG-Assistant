from unittest.mock import MagicMock


def _make_config():
    config = MagicMock()
    config.rag = MagicMock()
    config.rag.context_limit = 5
    config.rag.web_fallback_min_chunks = 3
    config.rag.web_fallback_hard_threshold = 0.7
    config.rag.web_fallback_soft_threshold = 0.85
    config.prompts = MagicMock()
    config.prompts.rag_user_template = (
        "{response_contract}\n\n"
        "Nguon:\n{context}\n\n"
        "Cau hoi:\n{question}"
    )
    config.prompts.user_template = "Context:\n{context}\n\nQuestion: {question}"
    config.prompts.rag_text_response_contract = "Tra loi bang markdown voi header `##`."
    return config


def _make_state():
    return {
        "query": "Cau hoi goc",
        "raw_query": "Cau hoi goc",
        "resolved_query": "Cau hoi da lam ro",
        "final_context": [
            {
                "context_id": "hybrid:1",
                "title": "GDP",
                "source": "src.com",
                "url": "https://example.com/gdp",
                "text": "GDP tang",
                "score": 0.9,
                "original_rank": 0,
                "source_type": "hybrid",
            }
        ],
        "reranked_docs": [],
        "answer": "Tra loi [[cite:hybrid:1]]",
        "generation_prompt": "prompt goc",
    }


def test_should_add_web_fallback_for_time_sensitive_query():
    from orchestrator.pipeline.rag_policy import should_add_web_fallback

    config = _make_config()
    state = _make_state() | {
        "reranked_docs": [{"index": 0, "score": 0.8}, {"index": 1, "score": 0.79}],
        "resolved_query": "Gia vang hom nay the nao?",
    }

    assert should_add_web_fallback(state, config) is True


def test_build_generation_prompt_uses_modular_prompt_parts():
    from orchestrator.pipeline.rag_prompts import build_generation_prompt

    config = _make_config()
    state = _make_state()

    prompt = build_generation_prompt(state, config)

    assert "Tra loi bang markdown voi header `##`." in prompt
    assert "Context ID: hybrid:1" in prompt
    assert "Cau hoi da lam ro" in prompt


def test_finalize_citations_appends_footer_and_normalizes_output():
    from orchestrator.pipeline.rag_context import finalize_citations

    state = _make_state() | {
        "citation_pool": {
            "hybrid:1": {
                "context_id": "hybrid:1",
                "title": "GDP",
                "url": "https://example.com/gdp",
                "source": "src.com",
                "score": 0.9,
                "original_rank": 0,
                "source_type": "hybrid",
            }
        }
    }

    result = finalize_citations(state, citation_limit=5)

    assert result["citations"] == [
        {
            "context_id": "hybrid:1",
            "title": "GDP",
            "url": "https://example.com/gdp",
            "source": "src.com",
            "source_type": "hybrid",
            "score": 0.9,
        }
    ]
    assert "[[cite:" not in result["answer"]
    assert "### Nguồn trích dẫn" in result["answer"]
