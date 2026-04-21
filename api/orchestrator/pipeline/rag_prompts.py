from __future__ import annotations

from collections.abc import Iterable

from orchestrator.config import PromptsConfig

DEFAULT_PROMPTS = PromptsConfig()
_CITATION_GUIDANCE = (
    "Hướng dẫn trích dẫn:\n"
    "- Trả lời bằng ngôn ngữ được sử dụng trong câu hỏi.\n"
    "- Trích dẫn inline ngay sau mỗi khẳng định bằng format **[S1]**, **[S2]**, v.v.\n"
    "- Có thể ghép nhiều nguồn liên tiếp: **[S1]** **[S2]**.\n"
    "- Không tạo nguồn ngoài danh sách đã cung cấp.\n"
)


def _raw_query(state: dict) -> str:
    return state.get("raw_query") or state.get("query", "")


def _resolved_query(state: dict) -> str:
    return state.get("resolved_query") or _raw_query(state)


def resolve_prompt_text(prompts, primary_name: str) -> str:
    primary_value = getattr(prompts, primary_name, None)
    if isinstance(primary_value, str) and primary_value.strip():
        return primary_value
    fallback_value = getattr(DEFAULT_PROMPTS, primary_name, None)
    if isinstance(fallback_value, str) and fallback_value.strip():
        return fallback_value
    return ""


def resolve_rag_system_prompt(prompts) -> str:
    return resolve_prompt_text(prompts, "rag_system_prompt")


def _resolve_response_contract(prompts) -> str:
    contract = resolve_prompt_text(prompts, "rag_text_response_contract")
    return contract or DEFAULT_PROMPTS.rag_text_response_contract


def _message_role(message) -> str:
    role = getattr(message, "role", None)
    if role:
        return str(role).upper()
    if isinstance(message, dict):
        return str(message.get("role", "user")).upper()
    return "USER"


def _message_content(message) -> str:
    if hasattr(message, "text_content"):
        return str(message.text_content()).strip()
    if isinstance(message, dict):
        return str(message.get("content", "")).strip()
    return ""


def serialize_conversation(messages: Iterable[object]) -> str:
    lines = []
    for message in messages:
        content = _message_content(message)
        if not content:
            continue
        lines.append(f"{_message_role(message)}: {content}")
    return "\n".join(lines)


def build_intent_prompt(
    messages: Iterable[object], prompts, *, image_caption: str = ""
) -> tuple[str, str]:
    transcript = serialize_conversation(messages)
    system_prompt = resolve_prompt_text(prompts, "intent_system_prompt")

    if image_caption:
        with_image_tpl = resolve_prompt_text(prompts, "intent_user_template_with_image")
        if with_image_tpl and "{image_caption}" in with_image_tpl:
            user_prompt = with_image_tpl.format(messages=transcript, image_caption=image_caption)
        else:
            # Fallback: prepend caption to the transcript line
            augmented = f"[Ảnh đính kèm: {image_caption}]\n{transcript}" if transcript else f"[Ảnh đính kèm: {image_caption}]"
            base_tpl = resolve_prompt_text(prompts, "intent_user_template")
            user_prompt = base_tpl.format(messages=augmented)
    else:
        base_tpl = resolve_prompt_text(prompts, "intent_user_template")
        user_prompt = base_tpl.format(messages=transcript)

    return system_prompt, user_prompt


def build_direct_prompt(
    *,
    messages: Iterable[object],
    resolved_query: str,
    prompts,
) -> str:
    response_contract = resolve_prompt_text(prompts, "direct_response_contract")
    conversation = serialize_conversation(messages) or f"USER: {resolved_query}"
    user_template = resolve_prompt_text(prompts, "direct_user_template")
    return user_template.format(
        response_contract=response_contract,
        conversation=conversation,
        question=resolved_query,
    )


def build_direct_prompt_with_web(
    *,
    messages: Iterable[object],
    resolved_query: str,
    web_results: list[dict],
    prompts,
) -> str:
    """Build user prompt for direct chat augmented with web search results.

    Returns a prompt that includes web context with [S1], [S2] IDs so the LLM
    can cite sources. Use with rag_system_prompt (not direct_system_prompt).
    """
    context_lines = []
    for i, item in enumerate(web_results):
        sid = f"S{i + 1}"
        title = item.get("title") or "Nguồn"
        url = item.get("url") or ""
        text = item.get("text") or ""
        header = f"[{sid}] {title}"
        if url:
            header += f" | {url}"
        context_lines.append(f"{header}\nContent: {text}")

    context_block = "\n\n".join(context_lines)
    conversation = serialize_conversation(messages) or f"USER: {resolved_query}"
    response_contract = resolve_prompt_text(prompts, "rag_text_response_contract")
    user_template = resolve_prompt_text(prompts, "rag_user_template")
    retrieved_context = (
        f"{_CITATION_GUIDANCE}\n\n{context_block}" if context_block else ""
    )

    if "{response_contract}" in user_template:
        return user_template.format(
            response_contract=response_contract,
            context=retrieved_context,
            question=resolved_query,
        )

    return (
        f"{response_contract}\n\n"
        f"{retrieved_context}\n\n"
        f"Hội thoại gần đây:\n{conversation}\n\n"
        f"Câu hỏi đã làm rõ:\n{resolved_query}"
    )


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
    return f"{_CITATION_GUIDANCE}\n\nNguồn đã gán ID:\n{context_block}"


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
            header += f" — {source}"
        # Include cite format so LLM outputs clickable markdown links directly
        cite_ref = f"[\\[{sid}\\]]({url})" if url else f"[{sid}]"
        parts.append(f"{header}\nCite: {cite_ref}\nContent: {text}")
    return "\n\n".join(parts)


def build_generation_prompt(state: dict, config) -> str:
    prompts = config.prompts
    response_contract = _resolve_response_contract(prompts)
    raw_context = _render_context_block(state, config.rag.context_limit)
    retrieved_context = f"{_CITATION_GUIDANCE}\n\n{raw_context}" if raw_context else ""
    user_template = resolve_prompt_text(prompts, "rag_user_template")

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
