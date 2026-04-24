from __future__ import annotations

from orchestrator.models.schemas import Message


def test_extract_latest_user_query_returns_last_user_message():
    from orchestrator.services.conversation import extract_latest_user_query

    messages = [
        Message(role="system", content="You are helpful."),
        Message(role="user", content="Bất động sản thế nào?"),
        Message(role="assistant", content="Đang phục hồi chậm."),
        Message(role="user", content="Còn trái phiếu thì sao?"),
    ]

    assert extract_latest_user_query(messages) == "Còn trái phiếu thì sao?"


def test_extract_image_contents_returns_urls_from_latest_user_message():
    from orchestrator.models.schemas import (
        Message, TextContentPart, ImageContentPart, ImageUrl,
    )
    from orchestrator.services.conversation import extract_image_contents

    messages = [
        Message(role="user", content="earlier text message"),
        Message(role="assistant", content="ok"),
        Message(
            role="user",
            content=[
                TextContentPart(text="What is this?"),
                ImageContentPart(image_url=ImageUrl(url="data:image/png;base64,abc")),
                ImageContentPart(image_url=ImageUrl(url="https://example.com/img.jpg")),
            ],
        ),
    ]

    result = extract_image_contents(messages)
    assert result == ["data:image/png;base64,abc", "https://example.com/img.jpg"]


def test_extract_image_contents_returns_empty_for_text_only():
    from orchestrator.models.schemas import Message
    from orchestrator.services.conversation import extract_image_contents

    messages = [Message(role="user", content="plain text")]
    assert extract_image_contents(messages) == []


def test_extract_image_contents_only_checks_latest_user_message():
    from orchestrator.models.schemas import (
        Message, ImageContentPart, ImageUrl,
    )
    from orchestrator.services.conversation import extract_image_contents

    messages = [
        Message(
            role="user",
            content=[ImageContentPart(image_url=ImageUrl(url="https://old.com/img.jpg"))],
        ),
        Message(role="assistant", content="response"),
        Message(role="user", content="follow-up text only"),
    ]

    # Latest user message is text-only; images in earlier messages are ignored
    assert extract_image_contents(messages) == []
