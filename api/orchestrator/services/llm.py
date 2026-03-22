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
    ):
        self._model = model
        self._temperature = temperature
        self._max_tokens = max_tokens
        self._client = AsyncOpenAI(base_url=url, api_key="fake", timeout=timeout)

    @traceable(name="Generate Answer", run_type="llm")
    async def generate(self, system_prompt: str, user_prompt: str) -> str:
        """Generate response from remote LLM. Returns Vietnamese error message on failure."""
        t0 = time.monotonic()
        try:
            response = await self._client.chat.completions.create(
                model=self._model,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt},
                ],
                temperature=self._temperature,
                max_tokens=self._max_tokens,
                extra_body={"chat_template_kwargs": {"enable_thinking": False}},
            )
            content = response.choices[0].message.content
            tokens = response.usage.completion_tokens if response.usage else 0
            latency_ms = int((time.monotonic() - t0) * 1000)
            logger.log(
                "LLM",
                "model={} tokens={} latency_ms={}",
                self._model,
                tokens,
                latency_ms,
            )
            return content
        except Exception as e:
            logger.error("LLM generation error: {}", e)
            return "Xin lỗi, không thể tạo phản hồi."
