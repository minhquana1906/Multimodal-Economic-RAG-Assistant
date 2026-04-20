from __future__ import annotations


class InferenceClient:
    """Store connection settings for the consolidated inference service."""

    def __init__(self, base_url: str, timeout: float):
        self.base_url = base_url.rstrip("/")
        self.timeout = timeout
