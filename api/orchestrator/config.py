from __future__ import annotations

from functools import lru_cache

from pydantic import BaseModel
from pydantic_settings import BaseSettings, SettingsConfigDict


class LLMConfig(BaseModel):
    url: str = "http://localhost:8004"
    model: str = "Qwen/Qwen3.5-4B"
    temperature: float = 0.7
    max_tokens: int = 512
    timeout: float = 60.0


class ServicesConfig(BaseModel):
    embedding_url: str = "http://embedding:8001"
    embedding_model: str = "Qwen/Qwen3-Embedding-0.6B"
    embedding_timeout: float = 15.0

    reranker_url: str = "http://reranker:8002"
    reranker_model: str = "Qwen/Qwen3-Reranker-0.6B"
    reranker_timeout: float = 15.0

    guard_url: str = "http://guard:8003"
    guard_model: str = "Qwen/Qwen3Guard-Gen-0.6B"
    guard_timeout: float = 10.0

    qdrant_url: str = "http://qdrant:6333"
    qdrant_collection: str = "econ_vn_news"

    # ASR Service (on-demand, profile: audio)
    asr_url: str = ""
    asr_timeout: float = 30.0
    asr_max_duration_s: int = 60


class RAGConfig(BaseModel):
    retrieval_top_k: int = 20
    rerank_top_n: int = 5
    fallback_min_chunks: int = 3
    fallback_score_threshold: float = 0.5
    context_limit: int = 5
    citation_limit: int = 5


class PromptsConfig(BaseModel):
    system_prompt: str = (
        "Bạn là trợ lý AI chuyên về kinh tế tài chính Việt Nam. "
        "Hãy trả lời ngắn gọn, chính xác dựa trên thông tin được cung cấp."
    )
    user_template: str = (
        "Dựa vào các đoạn văn bản sau:\n{context}\n\nTrả lời: {question}"
    )
    reranker_instruction: str = (
        "Cho một câu hỏi về kinh tế, tài chính, "
        "đánh giá mức độ liên quan của đoạn văn bản với câu hỏi"
    )
    no_context_message: str = "Xin lỗi, tôi không tìm thấy thông tin liên quan."
    guard_error_message: str = "Xin lỗi, yêu cầu của bạn không thể xử lý."
    apology_message: str = (
        "Xin lỗi, tôi không thể trả lời câu hỏi này theo nội dung của chúng tôi."
    )


class ObservabilityConfig(BaseModel):
    log_level: str = "INFO"
    langsmith_api_key: str | None = None
    langsmith_project: str = "multimodal-economic-rag"
    tavily_api_key: str | None = None


class Settings(BaseSettings):
    llm: LLMConfig = LLMConfig()
    services: ServicesConfig = ServicesConfig()
    rag: RAGConfig = RAGConfig()
    prompts: PromptsConfig = PromptsConfig()
    observability: ObservabilityConfig = ObservabilityConfig()

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        env_nested_delimiter="__",
        extra="ignore",
    )


@lru_cache()
def get_settings() -> Settings:
    return Settings()
