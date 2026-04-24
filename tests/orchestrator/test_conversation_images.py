"""Tests for multimodal conversation service helpers."""
from __future__ import annotations

import base64
import io

from orchestrator.models.schemas import ImageContentPart, ImageURL, Message
from orchestrator.services.conversation import (
    extract_latest_user_images,
    extract_latest_user_query,
    normalize_messages,
)


def _img_part(url: str = "data:image/png;base64,abc123") -> ImageContentPart:
    return ImageContentPart(image_url=ImageURL(url=url))


def _img_msg(text: str = "", url: str = "data:image/png;base64,abc123") -> Message:
    content: list = [{"type": "image_url", "image_url": {"url": url}}]
    if text:
        content.append({"type": "text", "text": text})
    return Message.model_validate({"role": "user", "content": content})


class TestNormalizeMessages:
    def test_text_only_collapsed_to_string(self):
        msgs = normalize_messages([Message(role="user", content="hello")])
        assert msgs[0].content == "hello"

    def test_image_message_preserved(self):
        msg = _img_msg()
        msgs = normalize_messages([msg])
        assert len(msgs) == 1
        assert msgs[0].has_images()

    def test_empty_text_only_dropped(self):
        msgs = normalize_messages([Message(role="user", content="   ")])
        assert msgs == []

    def test_image_only_not_dropped(self):
        msg = _img_msg()  # no text
        msgs = normalize_messages([msg])
        assert len(msgs) == 1

    def test_mixed_text_and_image_preserved(self):
        msg = _img_msg(text="describe this")
        msgs = normalize_messages([msg])
        assert msgs[0].has_images()
        assert msgs[0].text_content() == "describe this"


class TestExtractLatestUserImages:
    def test_returns_empty_for_text_only(self):
        msgs = [Message(role="user", content="hello")]
        assert extract_latest_user_images(msgs) == []

    def test_returns_image_parts_from_latest_user_msg(self):
        msgs = [_img_msg("q1"), Message(role="assistant", content="a1"), _img_msg("q2")]
        imgs = extract_latest_user_images(msgs)
        assert len(imgs) == 1
        assert isinstance(imgs[0], ImageContentPart)

    def test_ignores_assistant_images(self):
        assistant_with_img = Message.model_validate({
            "role": "assistant",
            "content": [{"type": "image_url", "image_url": {"url": "data:image/png;base64,x"}}],
        })
        msgs = [Message(role="user", content="hi"), assistant_with_img]
        assert extract_latest_user_images(msgs) == []

    def test_image_only_message_returns_parts(self):
        msg = _img_msg()
        assert len(extract_latest_user_images([msg])) == 1


class TestExtractLatestUserQuery:
    def test_returns_text_from_image_message(self):
        msg = _img_msg(text="analyse chart")
        assert extract_latest_user_query([msg]) == "analyse chart"

    def test_returns_empty_for_image_only(self):
        msg = _img_msg()  # no text
        assert extract_latest_user_query([msg]) == ""
