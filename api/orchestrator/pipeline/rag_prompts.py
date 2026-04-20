from __future__ import annotations


TEXT_RESPONSE_INSTRUCTIONS = """
Yêu cầu định dạng câu trả lời:
- Trả lời bằng tiếng Việt, câu trả lời lý tưởng là khoảng 1024 tokens (~2500-3000 ký tự).
- Mặc định chia câu trả lời thành 2-4 phần chính ở định dạng markdown với header `##`; tiêu đề do bạn tự đặt theo nội dung.
- Ngăn cách các phần bằng một dòng `---` để bố cục rõ ràng hơn.
- Ưu tiên dùng gạch đầu dòng khi đang liệt kê, dùng bảng khi cần so sánh, dùng đoạn văn khi giải thích hoặc mô tả.
- Sử dụng giọng điệu ấm áp, lễ phép, không lan man, không khẳng định quá mức, không tạo mục rỗng không cần thiết.
""".strip()

_CITATION_INSTRUCTION = (
    "Hướng dẫn trích dẫn: Trích dẫn inline bằng [S1], [S2], ... "
    "ngay sau mỗi khẳng định có căn cứ. Có thể dùng nhiều ID liên tiếp: [S1][S2]. "
    "Không tạo nguồn ngoài danh sách đã cung cấp."
)


def _raw_query(state: dict) -> str:
    return state.get("raw_query") or state.get("query", "")


def _resolved_query(state: dict) -> str:
    return state.get("resolved_query") or _raw_query(state)


def resolve_prompt_text(
    prompts, primary_name: str, fallback_name: str | None = None
) -> str:
    primary_value = getattr(prompts, primary_name, None)
    if isinstance(primary_value, str) and primary_value.strip():
        return primary_value
    if fallback_name is not None:
        fallback_value = getattr(prompts, fallback_name, None)
        if isinstance(fallback_value, str) and fallback_value.strip():
            return fallback_value
    return ""


def resolve_rag_system_prompt(prompts) -> str:
    return resolve_prompt_text(prompts, "rag_system_prompt", "system_prompt")


def _resolve_response_contract(prompts) -> str:
    contract = resolve_prompt_text(prompts, "rag_text_response_contract")
    return contract or TEXT_RESPONSE_INSTRUCTIONS


def build_rag_prompt(sources: list[dict]) -> str:
    """Build a formatted context block with inline [S1], [S2], ... citation IDs.

    Each source dict must have a 'source_id' key (e.g. 'S1') plus optional
    'title', 'url', and 'text' fields. Returns the full prompt string including
    citation instructions.
    """
    lines = []
    for src in sources:
        sid = src.get("source_id", "")
        title = src.get("title", "") or "Nguồn"
        url = src.get("url", "")
        text = src.get("text", "")
        header = f"[{sid}] {title}"
        if url:
            header = f"{header} | {url}"
        lines.append(f"{header}\nContent: {text}")

    context_block = "\n\n".join(lines)
    citation_instructions = (
        "Hướng dẫn trích dẫn:\n"
        "- Trả lời bằng tiếng Việt.\n"
        "- Trích dẫn inline bằng [S1], [S2], ... sau mỗi khẳng định có căn cứ.\n"
        "- Có thể dùng nhiều ID liên tiếp: [S1][S2].\n"
        "- Không tạo nguồn ngoài danh sách đã cung cấp.\n"
        "- Không thêm danh sách nguồn cuối câu trả lời."
    )
    return f"{citation_instructions}\n\nNguồn đã gán ID:\n{context_block}"


def _render_context_block(state: dict, context_limit: int) -> str:
    parts = []
    for i, item in enumerate(state.get("final_context", [])[:context_limit]):
        sid = f"S{i + 1}"
        title = item.get("title") or "N/A"
        source = item.get("source") or ""
        url = item.get("url") or ""
        text = item.get("text") or ""
        header = f"[{sid}] {title}"
        if source:
            header += f" | {source}"
        if url:
            header += f" | {url}"
        parts.append(f"{header}\nContent: {text}")
    return "\n\n".join(parts)


def build_generation_prompt(state: dict, config) -> str:
    prompts = config.prompts
    response_contract = _resolve_response_contract(prompts)
    raw_context = _render_context_block(state, config.rag.context_limit)
    retrieved_context = (
        f"{_CITATION_INSTRUCTION}\n\n{raw_context}" if raw_context else ""
    )
    user_template = resolve_prompt_text(prompts, "rag_user_template", "user_template")

    if "{response_contract}" in user_template:
        return user_template.format(
            response_contract=response_contract,
            context=retrieved_context,
            question=_resolved_query(state),
        )

    legacy_sections = [response_contract]
    if retrieved_context:
        legacy_sections.append(retrieved_context)
    return user_template.format(
        context="\n\n".join(section for section in legacy_sections if section),
        question=_resolved_query(state),
    )
