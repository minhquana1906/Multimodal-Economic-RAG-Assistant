from __future__ import annotations


TEXT_RESPONSE_INSTRUCTIONS = """
Yêu cầu định dạng câu trả lời:
- Trả lời bằng tiếng Việt.
- Trình bày bằng markdown rõ ràng, tự nhiên, dễ đọc.
- Mặc định chia câu trả lời thành 2-4 phần chính với header `##`; tiêu đề do bạn tự đặt theo nội dung thay vì dùng mẫu cố định.
- Ngăn cách các phần bằng một dòng `---` để bố cục rõ ràng hơn.
- Trong từng phần, ưu tiên văn xuôi tự nhiên và giải thích chi tiết, rõ ràng hơn một chút so với trả lời quá ngắn.
- Ưu tiên dùng gạch đầu dòng khi đang liệt kê ý, điều kiện, tác động, hoặc đối chiếu nguồn; không lạm dụng bullet point nếu đoạn văn sẽ tự nhiên hơn.
- Khi cần đối chiếu nguồn hoặc nêu giới hạn dữ liệu, hãy dành một phần riêng với header phù hợp do bạn tự đặt.
- Văn phong học thuật nhưng dễ hiểu với người dùng phổ thông.
- Không lan man, không khẳng định quá mức, không tạo mục rỗng không cần thiết.
""".strip()
AUDIO_RESPONSE_INSTRUCTIONS = """
Yêu cầu định dạng câu trả lời:
- Trả lời bằng tiếng Việt trong một đoạn ngắn.
- Dùng văn nói tự nhiên, ngắn gọn, trực tiếp vào ý chính.
- Giữ giọng văn nói tự nhiên, ấm áp, nhẹ nhàng.
- Chỉ nên dài 1-3 câu ngắn.
- Không dùng markdown, bullet point, header, hay danh sách.
- Không mở đầu vòng vo, không kết thúc bằng câu mời gọi kéo dài hội thoại.
""".strip()


def _raw_query(state: dict) -> str:
    return state.get("raw_query") or state.get("query", "")


def _resolved_query(state: dict) -> str:
    return state.get("resolved_query") or _raw_query(state)


def _response_mode(state: dict) -> str:
    return "audio" if state.get("response_mode") == "audio" else "text"


def resolve_prompt_text(prompts, primary_name: str, fallback_name: str | None = None) -> str:
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


def _resolve_response_contract(prompts, response_mode: str) -> str:
    if response_mode == "audio":
        contract = resolve_prompt_text(prompts, "rag_audio_response_contract")
        return contract or AUDIO_RESPONSE_INSTRUCTIONS

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
    return "\n\n".join(
        (
            f"Context ID: {item.get('context_id', '')}\n"
            f"Title: {item.get('title', 'N/A')}\n"
            f"Source: {item.get('source', '')}\n"
            f"URL: {item.get('url', '')}\n"
            f"Content: {item.get('text', '')}"
        )
        for item in state.get("final_context", [])[:context_limit]
    )


def build_generation_prompt(state: dict, config) -> str:
    prompts = config.prompts
    conversation_context = state.get("conversation_context", "").strip()
    context_block = conversation_context or "Ngữ cảnh hội thoại:\nKhông có."
    response_contract = _resolve_response_contract(prompts, _response_mode(state))
    retrieved_context = _render_context_block(state, config.rag.context_limit)
    user_template = resolve_prompt_text(prompts, "rag_user_template", "user_template")

    if "{conversation_context}" in user_template or "{response_contract}" in user_template:
        return user_template.format(
            conversation_context=context_block,
            response_contract=response_contract,
            context=retrieved_context,
            question=_resolved_query(state),
        )

    legacy_sections = [context_block, response_contract]
    if retrieved_context:
        legacy_sections.append(retrieved_context)
    return user_template.format(
        context="\n\n".join(section for section in legacy_sections if section),
        question=_resolved_query(state),
    )
