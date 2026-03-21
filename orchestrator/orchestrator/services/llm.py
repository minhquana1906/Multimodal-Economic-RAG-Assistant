from openai import AsyncOpenAI
import logging
from langsmith import traceable

logger = logging.getLogger(__name__)

class LLMClient:
    def __init__(self, base_url: str, api_key: str = "fake", timeout: float = 60.0):
        self.client = AsyncOpenAI(base_url=base_url, api_key=api_key)
        self.timeout = timeout

    @traceable(name="Generate Answer", run_type="llm")
    async def generate(
        self,
        system_prompt: str,
        user_prompt: str,
        temperature: float = 0.7,
        max_tokens: int = 512,
    ) -> str:
        """Generate response from remote LLM. Returns Vietnamese error message on failure."""
        try:
            response = await self.client.chat.completions.create(
                model="Qwen/Qwen3.5-4B",
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt},
                ],
                temperature=temperature,
                max_tokens=max_tokens,
                extra_body={"chat_template_kwargs": {"enable_thinking": False}},
            )
            return response.choices[0].message.content
        except Exception as e:
            logger.error(f"LLM generation error: {e}")
            return "Xin lỗi, không thể tạo phản hồi."
