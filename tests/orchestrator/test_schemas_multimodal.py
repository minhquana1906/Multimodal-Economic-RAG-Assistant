"""Tests for multimodal Message/ImageContentPart schema extensions."""
from __future__ import annotations

import base64

import pytest
from pydantic import ValidationError

from orchestrator.models.schemas import (
    ImageContentPart,
    ImageURL,
    Message,
    TextContentPart,
)


def _b64_png() -> str:
    """Minimal 1×1 red PNG as base64."""
    import io
    from PIL import Image
    img = Image.new("RGB", (1, 1), (255, 0, 0))
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    return "data:image/png;base64," + base64.b64encode(buf.getvalue()).decode()


class TestImageSchemas:
    def test_image_content_part_defaults(self):
        part = ImageContentPart(image_url=ImageURL(url="https://example.com/img.png"))
        assert part.type == "image_url"
        assert part.image_url.detail == "auto"

    def test_image_url_data_uri(self):
        uri = _b64_png()
        part = ImageContentPart(image_url=ImageURL(url=uri))
        assert part.image_url.url.startswith("data:image/png")

    def test_discriminated_union_text(self):
        msg = Message.model_validate({
            "role": "user",
            "content": [{"type": "text", "text": "hello"}],
        })
        assert msg.text_content() == "hello"
        assert not msg.has_images()

    def test_discriminated_union_image(self):
        uri = _b64_png()
        msg = Message.model_validate({
            "role": "user",
            "content": [{"type": "image_url", "image_url": {"url": uri}}],
        })
        assert msg.has_images()
        assert len(msg.image_parts()) == 1
        assert msg.text_content() == ""

    def test_mixed_content(self):
        uri = _b64_png()
        msg = Message.model_validate({
            "role": "user",
            "content": [
                {"type": "image_url", "image_url": {"url": uri}},
                {"type": "text", "text": "Phân tích ảnh này"},
            ],
        })
        assert msg.has_images()
        assert msg.text_content() == "Phân tích ảnh này"
        assert len(msg.image_parts()) == 1

    def test_to_openai_content_text_only(self):
        msg = Message(role="user", content="hello")
        assert msg.to_openai_content() == "hello"

    def test_to_openai_content_with_image(self):
        uri = _b64_png()
        msg = Message.model_validate({
            "role": "user",
            "content": [{"type": "image_url", "image_url": {"url": uri}}],
        })
        oc = msg.to_openai_content()
        assert isinstance(oc, list)
        assert oc[0]["type"] == "image_url"

    def test_string_content_has_no_images(self):
        msg = Message(role="user", content="plain text")
        assert msg.image_parts() == []
        assert not msg.has_images()

    def test_unknown_type_rejected(self):
        with pytest.raises((ValidationError, Exception)):
            Message.model_validate({
                "role": "user",
                "content": [{"type": "video_url", "video_url": {"url": "..."}}],
            })
