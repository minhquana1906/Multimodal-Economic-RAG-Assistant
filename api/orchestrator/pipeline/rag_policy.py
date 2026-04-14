from __future__ import annotations

import re


_TIME_SENSITIVE_MARKERS = (
    "hôm nay",
    "hiện tại",
    "mới nhất",
    "gần đây",
    "cập nhật",
    "latest",
    "today",
    "current",
    "recent",
    "năm nay",
    "tháng này",
    "quý này",
)


def _tokenize(text: str) -> list[str]:
    return re.findall(r"\w+", text.lower())


def _raw_query(state: dict) -> str:
    return state.get("raw_query") or state.get("query", "")


def _resolved_query(state: dict) -> str:
    return state.get("resolved_query") or _raw_query(state)


def _is_time_sensitive_query(query: str) -> bool:
    normalized = " ".join(query.lower().split())
    return any(marker in normalized for marker in _TIME_SENSITIVE_MARKERS)


def _material_query_expansion(raw_query: str, resolved_query: str) -> bool:
    raw_tokens = _tokenize(raw_query)
    resolved_tokens = _tokenize(resolved_query)
    if not raw_tokens or not resolved_tokens:
        return False

    raw_token_set = set(raw_tokens)
    extra_tokens = [token for token in resolved_tokens if token not in raw_token_set]
    return len(extra_tokens) >= 4 and len(resolved_tokens) >= len(raw_tokens) + 4


def _has_shallow_internal_support(reranked_docs: list[dict], config) -> bool:
    if len(reranked_docs) < config.rag.web_fallback_min_chunks:
        return True
    if len(reranked_docs) < 2:
        return True

    top_score = float(reranked_docs[0].get("score", 0.0) or 0.0)
    second_score = float(reranked_docs[1].get("score", 0.0) or 0.0)
    return (
        second_score < config.rag.web_fallback_hard_threshold
        or (top_score - second_score) >= 0.12
    )


def should_add_web_fallback(state: dict, config) -> bool:
    reranked_docs = state.get("reranked_docs", [])
    if not reranked_docs:
        return True

    top_score = float(reranked_docs[0].get("score", 0.0) or 0.0)
    if top_score < config.rag.web_fallback_hard_threshold:
        return True
    if top_score >= config.rag.web_fallback_soft_threshold:
        return False
    if _has_shallow_internal_support(reranked_docs, config):
        return True
    if _is_time_sensitive_query(_resolved_query(state)):
        return True
    if _material_query_expansion(_raw_query(state), _resolved_query(state)):
        return True
    return False
