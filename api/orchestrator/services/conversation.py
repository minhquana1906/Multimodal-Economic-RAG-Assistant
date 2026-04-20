from __future__ import annotations

from typing import Iterable

from orchestrator.models.schemas import Message


def _content_to_text(message: Message) -> str:
    return message.text_content().strip()


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
