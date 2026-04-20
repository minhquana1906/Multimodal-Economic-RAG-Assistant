from __future__ import annotations

import time
from typing import TypedDict

import httpx
from langsmith import traceable
from loguru import logger


class GuardResult(TypedDict):
    label: str
    safe_label: str | None
    categories: list[str]
    refusal: str | None


_DEFAULT_GUARD_RESULT: GuardResult = {
    "label": "unsafe",
    "safe_label": "Unsafe",
    "categories": [],
    "refusal": None,
}


class GuardClient:
    def __init__(self, url: str, timeout: float):
        self.url = url
        self.timeout = timeout

    def _normalize_result(self, payload: dict) -> GuardResult:
        label = payload.get("label", "unsafe")
        safe_label = payload.get("safe_label") or ("Safe" if label == "safe" else "Unsafe")
        categories = payload.get("categories") or []
        refusal = payload.get("refusal")
        return {
            "label": label,
            "safe_label": safe_label,
            "categories": categories,
            "refusal": refusal,
        }

    @traceable(name="Check Input Safety", run_type="chain")
    async def check_input(self, text: str) -> GuardResult:
        """Check if input is safe. Fail-closed on error."""
        t0 = time.monotonic()
        try:
            async with httpx.AsyncClient() as client:
                response = await client.post(
                    f"{self.url}/classify",
                    json={"text": text, "role": "input"},
                    timeout=self.timeout,
                )
                response.raise_for_status()
                result = self._normalize_result(response.json())
                latency_ms = int((time.monotonic() - t0) * 1000)
                logger.log(
                    "GUARD",
                    f"role=input label={result['label']} safe_label={result['safe_label']} categories={result['categories']} latency_ms={latency_ms}",
                )
                return result
        except Exception as e:
            logger.error(f"Guard service error: {e}")
            return dict(_DEFAULT_GUARD_RESULT)

    @traceable(name="Check Output Safety", run_type="chain")
    async def check_output(self, text: str, prompt: str) -> GuardResult:
        """Check if output is safe. Fail-closed on error."""
        t0 = time.monotonic()
        try:
            async with httpx.AsyncClient() as client:
                response = await client.post(
                    f"{self.url}/classify",
                    json={"text": text, "role": "output", "prompt": prompt},
                    timeout=self.timeout,
                )
                response.raise_for_status()
                result = self._normalize_result(response.json())
                latency_ms = int((time.monotonic() - t0) * 1000)
                logger.log(
                    "GUARD",
                    f"role=output label={result['label']} safe_label={result['safe_label']} categories={result['categories']} refusal={result['refusal']} latency_ms={latency_ms}",
                )
                return result
        except Exception as e:
            logger.error(f"Guard service error: {e}")
            return dict(_DEFAULT_GUARD_RESULT)
