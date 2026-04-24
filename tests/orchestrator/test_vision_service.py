"""Tests for orchestrator/services/vision.py."""
from __future__ import annotations

import base64
import io

import pytest

from orchestrator.models.schemas import ImageContentPart, ImageURL
from orchestrator.services.vision import (
    _decode_data_uri,
    _resize_if_needed,
    process_image_part,
    to_data_uri,
)


def _make_png_bytes(w: int = 100, h: int = 100) -> bytes:
    from PIL import Image
    img = Image.new("RGB", (w, h), color=(200, 100, 50))
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    return buf.getvalue()


def _make_data_uri(w: int = 100, h: int = 100) -> str:
    data = _make_png_bytes(w, h)
    return f"data:image/png;base64,{base64.b64encode(data).decode()}"


class TestDecodeDataUri:
    def test_decodes_correctly(self):
        data = _make_png_bytes()
        uri = f"data:image/png;base64,{base64.b64encode(data).decode()}"
        result_data, mime = _decode_data_uri(uri)
        assert mime == "image/png"
        assert result_data == data


class TestResizeIfNeeded:
    def test_no_resize_within_limits(self):
        data = _make_png_bytes(50, 50)
        result, mime = _resize_if_needed(data, "image/png", max_pixels=1_048_576, max_bytes=4_000_000)
        from PIL import Image
        img = Image.open(io.BytesIO(result))
        assert img.width == 50

    def test_resize_when_too_large(self):
        data = _make_png_bytes(2000, 2000)
        result, mime = _resize_if_needed(data, "image/png", max_pixels=1_048_576, max_bytes=4_000_000)
        from PIL import Image
        img = Image.open(io.BytesIO(result))
        assert img.width * img.height <= 1_048_576 * 1.05  # small tolerance

    def test_resize_when_bytes_too_large(self):
        data = _make_png_bytes(500, 500)
        # Force 1 byte limit → extreme resize
        result, _ = _resize_if_needed(data, "image/png", max_pixels=10_000_000, max_bytes=1000)
        assert len(result) <= 10000  # very small


class TestToDataUri:
    def test_round_trip(self):
        data = b"fake_image_bytes"
        uri = to_data_uri(data, "image/jpeg")
        assert uri.startswith("data:image/jpeg;base64,")
        decoded = base64.b64decode(uri.split(",")[1])
        assert decoded == data


class TestProcessImagePart:
    async def test_data_uri_processed(self):
        uri = _make_data_uri(50, 50)
        part = ImageContentPart(image_url=ImageURL(url=uri))
        result = await process_image_part(part)
        assert result["type"] == "image_url"
        assert result["image_url"]["url"].startswith("data:image/")

    async def test_oversized_image_is_resized(self):
        uri = _make_data_uri(2000, 2000)
        part = ImageContentPart(image_url=ImageURL(url=uri))
        result = await process_image_part(part, max_pixels=100 * 100)
        out_data = base64.b64decode(result["image_url"]["url"].split(",")[1])
        from PIL import Image
        img = Image.open(io.BytesIO(out_data))
        assert img.width * img.height <= 100 * 100 * 1.1

    async def test_invalid_url_fallback(self):
        bad_part = ImageContentPart(image_url=ImageURL(url="data:image/png;base64,NOT_VALID_BASE64!!!"))
        # Should not raise; falls back to original URL
        result = await process_image_part(bad_part)
        assert result["type"] == "image_url"
