from __future__ import annotations

import re
from typing import Iterable

from orchestrator.models.schemas import Message


DEFAULT_SUMMARIZE_THRESHOLD = 6
DEFAULT_RECENT_TURNS = 3
DEFAULT_PROMPT_TOKEN_BUDGET = 512
DEFAULT_RESERVED_COMPLETION_TOKENS = 256

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


def rewrite_followup_query(
    raw_query: str,
    summary: str,
    recent_turns: Iterable[Message],
) -> str:
    recent_turns = list(recent_turns)
    if not recent_turns or not _looks_like_followup(raw_query):
        return raw_query

    context_blocks: list[str] = []
    if summary:
        context_blocks.append(f"Tóm tắt trước đó: {summary}")

    recent_history = _format_history_lines(recent_turns)
    if recent_history:
        context_blocks.append(f"Lịch sử gần đây:\n{recent_history}")

    context_text = "\n".join(context_blocks).strip()
    if not context_text:
        return raw_query

    return f"{raw_query}\n\nNgữ cảnh hội thoại liên quan:\n{context_text}"


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
