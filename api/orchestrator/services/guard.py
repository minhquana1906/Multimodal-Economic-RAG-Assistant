from __future__ import annotations

import time

import httpx
from langsmith import traceable
from loguru import logger


class GuardClient:
    def __init__(self, url: str, timeout: float):
        self.url = url
        self.timeout = timeout

    @traceable(name="Check Input Safety", run_type="chain")
    async def check_input(self, text: str) -> bool:
        """Check if input is safe. Returns True if safe. Fail-closed (returns False on error)."""
        t0 = time.monotonic()
        try:
            async with httpx.AsyncClient() as client:
                response = await client.post(
                    f"{self.url}/classify",
                    json={"text": text, "role": "input"},
                    timeout=self.timeout,
                )
                response.raise_for_status()
                label = response.json().get("label", "unsafe")
                latency_ms = int((time.monotonic() - t0) * 1000)
                logger.log(
                    "GUARD",
                    "role=input label={} latency_ms={}",
                    label,
                    latency_ms,
                )
                return label == "safe"
        except Exception as e:
            logger.error("Guard service error: {}", e)
            return False

    @traceable(name="Check Output Safety", run_type="chain")
    async def check_output(self, text: str, prompt: str) -> bool:
        """Check if output is safe. Fail-closed (returns False on error)."""
        t0 = time.monotonic()
        try:
            async with httpx.AsyncClient() as client:
                response = await client.post(
                    f"{self.url}/classify",
                    json={"text": text, "role": "output", "prompt": prompt},
                    timeout=self.timeout,
                )
                response.raise_for_status()
                label = response.json().get("label", "unsafe")
                latency_ms = int((time.monotonic() - t0) * 1000)
                logger.log(
                    "GUARD",
                    "role=output label={} latency_ms={}",
                    label,
                    latency_ms,
                )
                return label == "safe"
        except Exception as e:
            logger.error("Guard service error: {}", e)
            return False
