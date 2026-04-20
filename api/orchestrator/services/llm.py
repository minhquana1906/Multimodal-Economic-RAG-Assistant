from __future__ import annotations

import json
import time
from collections.abc import AsyncIterator

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
        resolved_api_key = api_key or "dummy"
        self._client = AsyncOpenAI(
            base_url=url,
            api_key=resolved_api_key,
            timeout=timeout,
        )

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

    async def detect_intent(
        self,
        *,
        system_prompt: str,
        user_prompt: str,
        fallback_query: str,
    ) -> dict[str, str]:
        """Return routing intent for the conversation.

        Returns {"route": "direct"|"rag", "resolved_query": "..."} and falls back to
        {"route": "rag", "resolved_query": fallback_query} on invalid JSON.
        """
        try:
            content, _, _ = await self._create_completion(
                [
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt},
                ],
                max_tokens=128,
            )
            raw = (content or "").strip()
            # Strip markdown code block if present
            if raw.startswith("```"):
                raw = raw.split("```")[1]
                if raw.startswith("json"):
                    raw = raw[4:]
                raw = raw.strip()
            parsed = json.loads(raw)
            route = parsed.get("route", "rag")
            if route not in ("direct", "rag"):
                route = "rag"
            resolved_query = parsed.get("resolved_query") or fallback_query
            return {"route": route, "resolved_query": resolved_query}
        except Exception as e:
            logger.warning(f"detect_intent failed, defaulting to rag: {e}")
            return {"route": "rag", "resolved_query": fallback_query}

    @traceable(name="Stream Answer", run_type="llm")
    async def stream_chat(self, messages: list[dict[str, str]]) -> AsyncIterator[str]:
        """Yield delta text chunks from provider-side streaming. Does not buffer."""
        t0 = time.monotonic()
        total_chars = 0
        try:
            stream = await self._client.chat.completions.create(
                model=self._model,
                messages=messages,
                temperature=self._temperature,
                max_tokens=self._max_tokens,
                extra_body={"chat_template_kwargs": {"enable_thinking": False}},
                stream=True,
            )
            async for chunk in stream:
                delta = chunk.choices[0].delta if chunk.choices else None
                if delta and delta.content:
                    total_chars += len(delta.content)
                    yield delta.content
            latency_ms = int((time.monotonic() - t0) * 1000)
            logger.log(
                "LLM",
                f"model={self._model} chars={total_chars} latency_ms={latency_ms} task=stream",
            )
        except Exception as e:
            logger.error(f"LLM streaming error: {e}")
            yield "Xin lỗi, không thể tạo phản hồi."

    async def warm_start(self) -> None:
        """Send a minimal request to reduce first-token latency after startup."""
        await self._create_completion(
            [{"role": "user", "content": "ping"}],
            max_tokens=1,
        )
