"""TTS Client — sends text to TTS service for speech synthesis.

Follows the project pattern: httpx async client, @traceable, fail-open.
Unlike ASRClient (which raises on failure), TTSClient returns None on error
because a failed synthesis allows graceful fallback to a text-only response.
"""

from __future__ import annotations

import time

import httpx
from langsmith import traceable
from loguru import logger


class TTSClient:
    def __init__(self, url: str, timeout: float):
        self.url = url
        self.timeout = timeout

    @traceable(name="TTS Synthesize", run_type="chain")
    async def synthesize(self, text: str, speed: float = 1.0) -> bytes | None:
        """Send text to TTS service, return WAV audio bytes.

        Returns None on error (graceful fallback to text-only response).
        """
        t0 = time.monotonic()
        try:
            async with httpx.AsyncClient() as client:
                response = await client.post(
                    f"{self.url}/synthesize",
                    json={"text": text, "speed": speed},
                    timeout=self.timeout,
                )
                response.raise_for_status()
                audio_bytes = response.content
                latency_ms = int((time.monotonic() - t0) * 1000)
                logger.log(
                    "RETRIEVAL",
                    f"TTS synthesize: chars={len(text)} speed={speed} latency={latency_ms}ms",
                )
                return audio_bytes
        except httpx.HTTPStatusError as e:
            detail = ""
            try:
                detail = e.response.json().get("detail", "")
            except Exception:
                pass
            logger.warning(f"TTS service error {e.response.status_code}: {detail}")
            return None
        except Exception as e:
            logger.warning(f"TTS service unreachable: {e}")
            return None

    @traceable(name="TTS Unload", run_type="chain")
    async def unload(self) -> None:
        """Request explicit model unload to free VRAM."""
        try:
            async with httpx.AsyncClient() as client:
                response = await client.post(
                    f"{self.url}/unload",
                    timeout=10.0,
                )
                response.raise_for_status()
                logger.info(f"TTS model unloaded: {response.json()}")
        except Exception as e:
            logger.warning(f"Failed to unload TTS model: {e}")
