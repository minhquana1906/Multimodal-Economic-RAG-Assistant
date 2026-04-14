from __future__ import annotations

import re


_CITATION_PATTERN = re.compile(r"\[\[cite:([a-zA-Z0-9:_-]+)\]\]")


def build_context_item(
    item: dict,
    *,
    source_type: str,
    retrieval_stage: str,
    original_rank: int,
    context_id: str,
) -> dict:
    return {
        "context_id": context_id,
        "text": item.get("text", ""),
        "source_type": source_type,
        "retrieval_stage": retrieval_stage,
        "original_rank": original_rank,
        "collection_name": item.get("collection_name", ""),
        "doc_type": item.get(
            "doc_type",
            "web_page" if source_type == "web" else "retrieved_chunk",
        ),
        "chunk_type": item.get(
            "chunk_type",
            "web_snippet" if source_type == "web" else "text_chunk",
        ),
        "modality": item.get("modality", "text"),
        "source_quality": item.get(
            "source_quality",
            "external" if source_type == "web" else "retrieved",
        ),
        "title": item.get("title", ""),
        "url": item.get("url", ""),
        "source": item.get("source", ""),
        "score": float(item.get("score", 0.0) or 0.0),
        "image_path": item.get("image_path", ""),
        "structured_data": item.get("structured_data", {}),
    }


def combine_context_sources(
    *,
    retrieved_docs: list[dict],
    reranked_docs: list[dict],
    web_results: list[dict],
) -> dict:
    reranked_context = []
    for rank, ranked_doc in enumerate(reranked_docs):
        index = ranked_doc.get("index", -1)
        if 0 <= index < len(retrieved_docs):
            merged = dict(retrieved_docs[index])
            merged["score"] = ranked_doc.get("score", merged.get("score", 0.0))
            reranked_context.append(
                build_context_item(
                    merged,
                    source_type="hybrid",
                    retrieval_stage="rerank",
                    original_rank=rank,
                    context_id=f"hybrid:{merged.get('id', rank)}",
                )
            )

    web_context = [
        build_context_item(
            web_item,
            source_type="web",
            retrieval_stage=web_item.get("retrieval_stage", "web_fallback"),
            original_rank=int(web_item.get("original_rank", rank)),
            context_id=web_item.get("context_id", f"web:{rank}"),
        )
        for rank, web_item in enumerate(web_results)
    ]

    final_context = web_context + reranked_context if web_context else reranked_context
    citation_pool = {
        item["context_id"]: item for item in final_context if item.get("context_id")
    }
    return {"final_context": final_context, "citation_pool": citation_pool}


def _citation_sort_key(item: dict) -> tuple[float, int]:
    return (
        float(item.get("score", 0.0) or 0.0),
        -int(item.get("original_rank", 0) or 0),
    )


def _normalize_citation(item: dict) -> dict:
    source = item.get("source", "") or item.get("url", "") or "unknown"
    return {
        "context_id": item.get("context_id", ""),
        "title": item.get("title", ""),
        "url": item.get("url", ""),
        "source": source,
        "source_type": item.get("source_type", ""),
        "score": float(item.get("score", 0.0) or 0.0),
    }


def _format_citation_line(item: dict) -> str:
    title = item.get("title", "") or "Nguồn tham khảo"
    url = item.get("url", "") or ""
    source = item.get("source", "") or url or "unknown"
    score = float(item.get("score", 0.0) or 0.0)
    title_part = f"[{title}]({url})" if url else title
    return f"- {title_part} - **{source} ({score:.4f})**"


def finalize_citations(state: dict, *, citation_limit: int) -> dict:
    citation_pool = state.get("citation_pool", {})
    candidates = sorted(
        citation_pool.values(),
        key=_citation_sort_key,
        reverse=True,
    )[:citation_limit]

    citations = [_normalize_citation(item) for item in candidates]
    answer = (state.get("answer") or "").rstrip()
    answer = _CITATION_PATTERN.sub("", answer)
    answer = re.sub(r"[ ]{2,}", " ", answer).strip()

    if citations and state.get("response_mode") != "audio":
        citation_lines = "\n".join(_format_citation_line(item) for item in candidates)
        answer = f"{answer}\n\n----\n\n### Nguồn trích dẫn\n{citation_lines}"

    return {"answer": answer, "citations": citations}
