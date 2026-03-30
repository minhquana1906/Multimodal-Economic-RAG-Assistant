from __future__ import annotations

import re
from typing import Any, Iterable

from loguru import logger

from orchestrator.models.schemas import Message


DEFAULT_SUMMARIZE_THRESHOLD = 6
DEFAULT_RECENT_TURNS = 3
DEFAULT_PROMPT_TOKEN_BUDGET = 512
DEFAULT_RESERVED_COMPLETION_TOKENS = 256
QUERY_REWRITE_INSTRUCTION = """
Bạn là bộ chuẩn hóa truy vấn cho hệ thống RAG.
Viết lại thành đúng 1 câu hỏi hoàn chỉnh bằng tiếng Việt.
Giữ nguyên ý định người dùng, sửa chính tả nếu cần, làm rõ đại từ tham chiếu bằng ngữ cảnh liên quan.
Không thêm thông tin mới ngoài hội thoại.
Chỉ trả về đúng câu hỏi đã được viết lại, không giải thích thêm.
""".strip()

_AUXILIARY_TASK_PATTERNS = {
    "title": (
        "generate a concise, 3-5 word title",
        "generate a concise 3-5 word title",
    ),
    "tags": (
        "generate 1-3 broad tags categorizing the main themes",
    ),
    "follow_ups": (
        "suggest 3-5 relevant follow-up questions",
    ),
}
_FOLLOW_UP_MARKERS = (
    "còn",
    "thì sao",
    "ra sao",
    "vậy",
    "thế",
    "nó",
    "điều này",
    "điều đó",
)
_CHAT_HISTORY_PATTERN = re.compile(
    r"<chat_history>\s*(?P<history>.*?)\s*</chat_history>",
    re.IGNORECASE | re.DOTALL,
)
_QUERY_NORMALIZATION_REPLACEMENTS: tuple[tuple[str, str], ...] = (
    (r"\bdn\b", "doanh nghiệp"),
    (r"\bttck\b", "thị trường chứng khoán"),
    (r"\bko\b", "không"),
    (r"\bkh\b", "khách hàng"),
    (r"\bnhư nào\b", "như thế nào"),
)


def _content_to_text(message: Message) -> str:
    return message.text_content().strip()


def _format_history_lines(messages: Iterable[Message]) -> str:
    lines = []
    for message in messages:
        if message.role == "system":
            continue
        content = _content_to_text(message)
        if not content:
            continue
        lines.append(f"{message.role.upper()}: {content}")
    return "\n".join(lines)


def _non_system_messages(messages: Iterable[Message]) -> list[Message]:
    return [message for message in messages if message.role != "system"]


def _estimate_tokens(text: str) -> int:
    return max(1, len(text.split()))


def _normalize_query_text(text: str) -> str:
    cleaned = " ".join(text.split())
    if not cleaned:
        return ""

    normalized = cleaned
    for pattern, replacement in _QUERY_NORMALIZATION_REPLACEMENTS:
        normalized = re.sub(pattern, replacement, normalized, flags=re.IGNORECASE)

    normalized = normalized[0].upper() + normalized[1:] if normalized else normalized
    if normalized and normalized[-1] not in ".?!":
        normalized = f"{normalized}?"
    return normalized


def _looks_like_followup(raw_query: str) -> bool:
    normalized = " ".join(raw_query.lower().split())
    if any(marker in normalized for marker in _FOLLOW_UP_MARKERS):
        return True
    return len(normalized.split()) <= 7


def normalize_messages(messages: Iterable[Message]) -> list[Message]:
    normalized: list[Message] = []
    for message in messages:
        if not isinstance(message, Message):
            message = Message.model_validate(message)
        content = _content_to_text(message)
        if not content:
            continue
        normalized.append(Message(role=message.role, content=content))
    return normalized


def extract_latest_user_query(messages: Iterable[Message]) -> str:
    for message in reversed(normalize_messages(messages)):
        if message.role == "user":
            return _content_to_text(message)
    return ""


def should_summarize(
    messages: Iterable[Message],
    summarize_threshold: int = DEFAULT_SUMMARIZE_THRESHOLD,
    prompt_token_budget: int = DEFAULT_PROMPT_TOKEN_BUDGET,
    reserved_completion_tokens: int = DEFAULT_RESERVED_COMPLETION_TOKENS,
) -> bool:
    non_system_messages = _non_system_messages(normalize_messages(messages))
    if len(non_system_messages) <= summarize_threshold:
        return False

    usable_budget = max(1, prompt_token_budget - reserved_completion_tokens)
    estimated_prompt_tokens = sum(
        _estimate_tokens(_content_to_text(message)) for message in non_system_messages
    )
    return estimated_prompt_tokens > usable_budget


def summarize_history(
    messages: Iterable[Message],
    keep_recent_turns: int = DEFAULT_RECENT_TURNS,
) -> str:
    non_system_messages = _non_system_messages(normalize_messages(messages))
    earlier_turns = non_system_messages[:-keep_recent_turns]
    if not earlier_turns:
        return ""
    return _format_history_lines(earlier_turns)


def build_conversation_context(summary: str, recent_turns: Iterable[Message]) -> str:
    sections: list[str] = []
    if summary:
        sections.append(f"Tóm tắt hội thoại:\n{summary}")

    recent_history = _format_history_lines(recent_turns)
    if recent_history:
        sections.append(f"Các lượt gần đây:\n{recent_history}")

    return "\n\n".join(sections)


def _dedupe_preserving_order(items: Iterable[str]) -> list[str]:
    seen: set[str] = set()
    unique_items: list[str] = []
    for item in items:
        normalized = item.strip()
        if not normalized:
            continue
        marker = normalized.casefold()
        if marker in seen:
            continue
        seen.add(marker)
        unique_items.append(normalized)
    return unique_items


def normalize_query_with_context(
    raw_query: str,
    summary: str,
    recent_turns: Iterable[Message],
) -> str:
    normalized_query = _normalize_query_text(raw_query)
    if not normalized_query:
        return ""

    recent_turns = list(recent_turns)
    context_fragments: list[str] = []
    if summary:
        context_fragments.append(summary)

    latest_user_turn = ""
    for message in reversed(recent_turns):
        if message.role != "user":
            continue
        candidate = _normalize_query_text(_content_to_text(message))
        if candidate and candidate.casefold() != normalized_query.casefold():
            latest_user_turn = candidate.rstrip("?")
            break

    if latest_user_turn:
        context_fragments.append(latest_user_turn)

    context_fragments = _dedupe_preserving_order(context_fragments)
    if not context_fragments:
        return normalized_query

    context_text = "; ".join(
        fragment.rstrip(".?!") for fragment in context_fragments if fragment.strip()
    )
    return f"{normalized_query} Ngữ cảnh liên quan: {context_text}."


def build_query_rewrite_prompt(
    raw_query: str,
    summary: str,
    recent_turns: Iterable[Message],
) -> str:
    sections = [QUERY_REWRITE_INSTRUCTION, f"Câu hỏi gốc:\n{raw_query.strip()}"]
    if summary:
        sections.append(f"Tóm tắt hội thoại liên quan:\n{summary.strip()}")

    recent_history = _format_history_lines(recent_turns)
    if recent_history:
        sections.append(f"Các lượt hội thoại gần đây:\n{recent_history}")

    sections.append(
        "Viết lại thành đúng 1 câu hỏi hoàn chỉnh, đúng chính tả, rõ nghĩa, phù hợp cho retrieval và web search."
    )
    return "\n\n".join(section for section in sections if section)


async def resolve_user_query(
    *,
    raw_query: str,
    summary: str,
    recent_turns: Iterable[Message],
    llm: Any | None,
    max_tokens: int | None = None,
) -> str:
    del max_tokens
    fallback_query = normalize_query_with_context(raw_query, summary, recent_turns)
    if not raw_query.strip():
        return fallback_query
    if llm is None:
        return fallback_query

    prompt = build_query_rewrite_prompt(raw_query, summary, recent_turns)
    try:
        rewritten_query = await llm.complete_prompt(prompt)
    except Exception as exc:
        logger.warning("Query rewrite failed, falling back to deterministic rewrite: {}", exc)
        return fallback_query or _normalize_query_text(raw_query)

    rewritten_query = _normalize_query_text(rewritten_query)
    if rewritten_query:
        return rewritten_query
    return fallback_query or _normalize_query_text(raw_query)


def rewrite_followup_query(
    raw_query: str,
    summary: str,
    recent_turns: Iterable[Message],
) -> str:
    recent_turns = list(recent_turns)
    if not recent_turns or not _looks_like_followup(raw_query):
        return _normalize_query_text(raw_query)
    return normalize_query_with_context(raw_query, summary, recent_turns)


def classify_task(messages: Iterable[Message], latest_user_message: str) -> str:
    del messages
    normalized = " ".join(latest_user_message.lower().split())
    if normalized.startswith("### task:"):
        for task_type, patterns in _AUXILIARY_TASK_PATTERNS.items():
            if any(pattern in normalized for pattern in patterns):
                return task_type
    return "chat"


def build_auxiliary_history(
    messages: Iterable[Message],
    latest_user_message: str,
) -> str:
    normalized_messages = normalize_messages(messages)
    if normalized_messages and normalized_messages[-1].role == "user":
        latest_content = _content_to_text(normalized_messages[-1])
        if latest_content == latest_user_message.strip():
            prior_turns = _non_system_messages(normalized_messages[:-1])
        else:
            prior_turns = _non_system_messages(normalized_messages)
    else:
        prior_turns = _non_system_messages(normalized_messages)

    if len(prior_turns) >= 2:
        return _format_history_lines(prior_turns)

    match = _CHAT_HISTORY_PATTERN.search(latest_user_message)
    if match:
        return match.group("history").strip()

    return _format_history_lines(prior_turns)


def build_auxiliary_prompt(task_type: str, history: str) -> str:
    instructions = {
        "title": "Generate a concise, 3-5 word title with an emoji summarizing the chat history.",
        "tags": (
            "Generate 1-3 broad tags categorizing the main themes of the chat history, "
            "along with 1-3 more specific subtopic tags."
        ),
        "follow_ups": "Suggest 3-5 relevant follow-up questions or prompts based on the chat history.",
    }
    instruction = instructions.get(task_type, "")
    if not instruction:
        return ""

    return (
        f"### Task:\n{instruction}\n\n"
        f"### Chat History:\n<chat_history>\n{history}\n</chat_history>"
    )
