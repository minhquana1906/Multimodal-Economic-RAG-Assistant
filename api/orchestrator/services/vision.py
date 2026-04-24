from __future__ import annotations

import base64
import io

import httpx
from loguru import logger


def _decode_data_uri(url: str) -> tuple[bytes, str]:
    header, b64data = url.split(",", 1)
    mime = header.split(";")[0].split(":")[1]
    return base64.b64decode(b64data), mime


async def _fetch_image_url(url: str) -> tuple[bytes, str]:
    async with httpx.AsyncClient(follow_redirects=True, timeout=10.0) as client:
        resp = await client.get(url)
        resp.raise_for_status()
    ct = resp.headers.get("content-type", "image/jpeg").split(";")[0]
    return resp.content, ct


def _resize_if_needed(data: bytes, mime: str, max_pixels: int, max_bytes: int) -> tuple[bytes, str]:
    try:
        from PIL import Image
    except ImportError:
        return data, mime

    img = Image.open(io.BytesIO(data))
    if img.mode not in ("RGB", "RGBA", "L"):
        img = img.convert("RGB")

    w, h = img.size
    if w * h > max_pixels:
        ratio = (max_pixels / (w * h)) ** 0.5
        img = img.resize((max(1, int(w * ratio)), max(1, int(h * ratio))), Image.LANCZOS)

    out_fmt = "PNG" if img.mode == "RGBA" else "JPEG"
    out_mime = "image/png" if out_fmt == "PNG" else "image/jpeg"
    buf = io.BytesIO()
    img.save(buf, format=out_fmt, **({} if out_fmt == "PNG" else {"quality": 85}))
    result = buf.getvalue()

    if len(result) > max_bytes:
        ratio = (max_bytes / len(result)) ** 0.5
        img = img.resize((max(1, int(img.width * ratio)), max(1, int(img.height * ratio))), Image.LANCZOS)
        buf = io.BytesIO()
        img.save(buf, format=out_fmt, **({} if out_fmt == "PNG" else {"quality": 75}))
        result = buf.getvalue()

    return result, out_mime


def to_data_uri(data: bytes, mime: str) -> str:
    return f"data:{mime};base64,{base64.b64encode(data).decode()}"


async def process_image_part(
    part,  # ImageContentPart
    *,
    max_pixels: int = 1_048_576,
    max_bytes: int = 4_000_000,
) -> dict:
    """Validate, resize if needed, and return an OpenAI-style image_url dict."""
    url = part.image_url.url
    try:
        if url.startswith("data:"):
            data, mime = _decode_data_uri(url)
        else:
            data, mime = await _fetch_image_url(url)
        data, mime = _resize_if_needed(data, mime, max_pixels, max_bytes)
        safe_url = to_data_uri(data, mime)
    except Exception as exc:
        logger.warning(f"image processing failed, using original URL: {exc}")
        safe_url = url
    return {"type": "image_url", "image_url": {"url": safe_url, "detail": part.image_url.detail}}
