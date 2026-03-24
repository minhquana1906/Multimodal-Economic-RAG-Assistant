from __future__ import annotations

import time

from langsmith import traceable
from loguru import logger
from openai import AsyncOpenAI


class LLMClient:
    def __init__(
        self,
        url: str,
        model: str,
        temperature: float,
        max_tokens: int,
        timeout: float,
        api_key: str = "",
    ):
        self._model = model
        self._temperature = temperature
        self._max_tokens = max_tokens
        self._client = AsyncOpenAI(base_url=url, api_key=api_key, timeout=timeout)

    async def _create_completion(
        self,
        messages: list[dict[str, str]],
        *,
        max_tokens: int | None = None,
    ) -> tuple[str | None, int, int]:
        t0 = time.monotonic()
        response = await self._client.chat.completions.create(
            model=self._model,
            messages=messages,
            temperature=self._temperature,
            max_tokens=max_tokens or self._max_tokens,
            extra_body={"chat_template_kwargs": {"enable_thinking": False}},
        )
        content = response.choices[0].message.content
        tokens = response.usage.completion_tokens if response.usage else 0
        latency_ms = int((time.monotonic() - t0) * 1000)
        return content, tokens, latency_ms

    @traceable(name="Generate Answer", run_type="llm")
    async def generate(self, system_prompt: str, user_prompt: str) -> str:
        """Generate response from remote LLM. Returns Vietnamese error message on failure."""
        try:
            content, tokens, latency_ms = await self._create_completion(
                [
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt},
                ]
            )
            logger.log(
                "LLM",
                f"model={self._model} tokens={tokens} latency_ms={latency_ms}",
            )
            return content or ""
        except Exception as e:
            logger.error(f"LLM generation error: {e}")
            return "Xin lỗi, không thể tạo phản hồi."

    async def complete_prompt(self, prompt: str, max_tokens: int | None = None) -> str:
        """Handle auxiliary UI prompts without entering the traced RAG path."""
        try:
            content, tokens, latency_ms = await self._create_completion(
                [{"role": "user", "content": prompt}],
                max_tokens=max_tokens,
            )
            logger.log(
                "LLM",
                f"model={self._model} tokens={tokens} latency_ms={latency_ms} task=aux",
            )
            return content or ""
        except Exception as e:
            logger.error(f"Auxiliary LLM generation error: {e}")
            return ""
