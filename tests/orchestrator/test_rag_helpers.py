from unittest.mock import MagicMock


def _make_config():
    config = MagicMock()
    config.rag = MagicMock()
    config.rag.context_limit = 5
    config.rag.web_fallback_min_chunks = 3
    config.rag.web_fallback_hard_threshold = 0.7
    config.rag.web_fallback_soft_threshold = 0.85
    config.prompts = MagicMock()
    config.prompts.intent_system_prompt = "Ban la bo dinh tuyen."
    config.prompts.intent_user_template = "Hoi thoai:\n{messages}"
    config.prompts.direct_system_prompt = "Tra loi truc tiep bang tieng Viet."
    config.prompts.direct_response_contract = (
        "Yeu cau dinh dang:\n"
        "- Chia thanh 2-4 phan voi header `##`.\n"
        "- Dung bang khi can so sanh.\n"
        "- Dung bullet cho y chinh."
    )
    config.prompts.direct_user_template = (
        "{response_contract}\n\n"
        "Hoi thoai gan day:\n{conversation}\n\n"
        "Yeu cau hien tai:\n{question}"
    )
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

    use_web, _ = should_add_web_fallback(state, config)
    assert use_web


def test_build_generation_prompt_uses_modular_prompt_parts():
    from orchestrator.pipeline.rag_prompts import build_generation_prompt

    config = _make_config()
    state = _make_state()

    prompt = build_generation_prompt(state, config)

    assert "Tra loi bang markdown voi header `##`." in prompt
    assert "[S1]" in prompt
    assert "GDP" in prompt
    assert "Cau hoi da lam ro" in prompt


def test_build_direct_prompt_uses_contract_and_serialized_conversation():
    from orchestrator.models.schemas import Message
    from orchestrator.pipeline.rag_prompts import build_direct_prompt

    config = _make_config()
    messages = [
        Message(role="user", content="Xin chao"),
        Message(role="assistant", content="Chao ban"),
        Message(role="user", content="Giai thich giup toi lam phat"),
    ]

    prompt = build_direct_prompt(
        messages=messages,
        resolved_query="Hay giai thich lam phat theo cach de hieu",
        prompts=config.prompts,
    )

    assert "header `##`" in prompt
    assert "Dung bang khi can so sanh" in prompt
    assert "USER: Xin chao" in prompt
    assert "ASSISTANT: Chao ban" in prompt
    assert "Hay giai thich lam phat theo cach de hieu" in prompt


def test_build_intent_prompt_uses_configured_template():
    from orchestrator.models.schemas import Message
    from orchestrator.pipeline.rag_prompts import build_intent_prompt

    config = _make_config()
    messages = [
        Message(role="user", content="GDP Viet Nam 2024 la bao nhieu?"),
    ]

    system_prompt, user_prompt = build_intent_prompt(messages, config.prompts)

    assert system_prompt == "Ban la bo dinh tuyen."
    assert "USER: GDP Viet Nam 2024 la bao nhieu?" in user_prompt
    assert user_prompt.startswith("Hoi thoai:")


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

    result = finalize_citations(state, context_limit=5, citation_limit=5)

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
