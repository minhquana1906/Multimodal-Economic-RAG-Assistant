from __future__ import annotations

from typing import Iterable

from orchestrator.models.schemas import ImageContentPart, Message


def normalize_messages(messages: Iterable[Message]) -> list[Message]:
    normalized: list[Message] = []
    for message in messages:
        if not isinstance(message, Message):
            message = Message.model_validate(message)
        has_text = bool(message.text_content().strip())
        has_images = message.has_images()
        if not has_text and not has_images:
            continue
        if has_text and not has_images:
            # Text-only: collapse to string (preserve existing behaviour)
            normalized.append(Message(role=message.role, content=message.text_content().strip()))
        else:
            # Has images (with or without text): preserve full content list
            normalized.append(message)
    return normalized


def extract_latest_user_query(messages: Iterable[Message]) -> str:
    for message in reversed(normalize_messages(list(messages))):
        if message.role == "user":
            return message.text_content().strip()
    return ""


def extract_latest_user_images(messages: Iterable[Message]) -> list[ImageContentPart]:
    """Return image parts from the most recent user message, or []."""
    for message in reversed(normalize_messages(list(messages))):
        if message.role == "user":
            return message.image_parts()
    return []


def extract_image_contents(messages: Iterable[Message]) -> list[str]:
    """Return image URLs from the latest user message's content parts."""
    validated = [m if isinstance(m, Message) else Message.model_validate(m) for m in messages]
    for message in reversed(validated):
        if message.role == "user":
            if isinstance(message.content, list):
                return [
                    part.image_url.url
                    for part in message.content
                    if isinstance(part, ImageContentPart)
                ]
            else:
                # Latest user message is found and it's text-only
                return []
    return []
