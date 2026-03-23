"""ASR Client — sends audio to ASR service for transcription.

Follows the project pattern: httpx async client, @traceable, fail-closed.
Unlike GuardClient (which fail-closes to False), ASRClient raises ASRError
because a failed transcription means the request cannot proceed.
"""

from __future__ import annotations

import time

import httpx
from langsmith import traceable
from loguru import logger


class ASRError(Exception):
    """Raised when ASR transcription fails."""


class ASRClient:
    def __init__(self, url: str, timeout: float):
        self.url = url
        self.timeout = timeout

    @traceable(name="ASR Transcribe", run_type="chain")
    async def transcribe(
        self, audio_bytes: bytes, language: str = "vi", content_type: str = "audio/wav"
    ) -> str:
        """Send audio to ASR service, return transcribed text.

        Raises ASRError on any failure (service down, decode error, empty result).
        """
        t0 = time.monotonic()
        try:
            async with httpx.AsyncClient() as client:
                response = await client.post(
                    f"{self.url}/transcribe",
                    files={"file": ("audio.wav", audio_bytes, content_type)},
                    data={"language": language},
                    timeout=self.timeout,
                )
                response.raise_for_status()
                data = response.json()
                text = data.get("text", "")
                latency_ms = int((time.monotonic() - t0) * 1000)
                logger.log(
                    "RETRIEVAL",
                    "ASR transcribe: lang={} duration={}s latency={}ms",
                    language,
                    data.get("duration_seconds", "?"),
                    latency_ms,
                )
                return text
        except httpx.HTTPStatusError as e:
            detail = ""
            try:
                detail = e.response.json().get("detail", "")
            except Exception:
                pass
            raise ASRError(f"ASR service error {e.response.status_code}: {detail}") from e
        except Exception as e:
            raise ASRError(f"ASR service unreachable: {e}") from e

    @traceable(name="ASR Unload", run_type="chain")
    async def unload(self) -> None:
        """Request explicit model unload to free VRAM before TTS."""
        try:
            async with httpx.AsyncClient() as client:
                response = await client.post(
                    f"{self.url}/unload",
                    timeout=10.0,
                )
                response.raise_for_status()
                logger.info("ASR model unloaded: {}", response.json())
        except Exception as e:
            logger.warning("Failed to unload ASR model: {}", e)
