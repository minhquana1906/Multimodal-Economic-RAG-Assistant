import httpx
import logging
from langsmith import traceable

logger = logging.getLogger(__name__)

class GuardClient:
    def __init__(self, url: str, timeout: float):
        self.url = url
        self.timeout = timeout

    @traceable(name="Check Input Safety", run_type="chain")
    async def check_input(self, text: str) -> bool:
        """Check if input is safe. Returns True if safe. Fail-closed (returns False on error)."""
        try:
            async with httpx.AsyncClient() as client:
                response = await client.post(
                    f"{self.url}/classify",
                    json={"text": text, "role": "input"},
                    timeout=self.timeout,
                )
                response.raise_for_status()
                return response.json().get("label") == "safe"
        except Exception as e:
            logger.error(f"Guard service error: {e}")
            return False

    @traceable(name="Check Output Safety", run_type="chain")
    async def check_output(self, text: str, prompt: str) -> bool:
        """Check if output is safe. Fail-closed (returns False on error)."""
        try:
            async with httpx.AsyncClient() as client:
                response = await client.post(
                    f"{self.url}/classify",
                    json={"text": text, "role": "output", "prompt": prompt},
                    timeout=self.timeout,
                )
                response.raise_for_status()
                return response.json().get("label") == "safe"
        except Exception as e:
            logger.error(f"Guard service error: {e}")
            return False
