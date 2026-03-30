from __future__ import annotations

from functools import lru_cache

from pydantic import AliasChoices, BaseModel, Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class LLMConfig(BaseModel):
    url: str
    model: str
    temperature: float
    max_tokens: int
    timeout: float
    api_key: str = ""


class ServicesConfig(BaseModel):
    embedding_url: str
    embedding_model: str
    embedding_timeout: float
    embedding_max_seq_length: int
    embedding_encode_batch_size: int

    reranker_url: str
    reranker_model: str
    reranker_timeout: float

    guard_url: str
    guard_model: str
    guard_timeout: float
    guard_max_new_tokens: int

    asr_url: str
    asr_model: str
    asr_timeout: float
    asr_max_duration_s: int
    asr_idle_timeout: int

    tts_url: str
    tts_model: str
    tts_timeout: float
    tts_speed: float
    tts_sample_rate: int
    tts_idle_timeout: int

    qdrant_url: str
    qdrant_collection: str


class RAGConfig(BaseModel):
    retrieval_top_k: int
    rerank_top_n: int
    web_fallback_min_chunks: int = Field(
        default=2,
        validation_alias=AliasChoices(
            "web_fallback_min_chunks",
            "fallback_min_chunks",
        ),
    )
    web_fallback_hard_threshold: float = Field(
        default=0.70,
        validation_alias=AliasChoices(
            "web_fallback_hard_threshold",
            "fallback_score_threshold",
        ),
    )
    web_fallback_soft_threshold: float = Field(
        default=0.85,
        validation_alias=AliasChoices(
            "web_fallback_soft_threshold",
        ),
    )
    context_limit: int
    citation_limit: int

    @property
    def fallback_min_chunks(self) -> int:
        return self.web_fallback_min_chunks

    @property
    def fallback_score_threshold(self) -> float:
        return self.web_fallback_hard_threshold


class PromptsConfig(BaseModel):
    system_prompt: str = (
        "Bạn là trợ lý AI chuyên về kinh tế tài chính Việt Nam. Hãy trả lời ngắn gọn, chính xác bằng tiếng Việt dựa trên nguồn được cung cấp."
    )
    user_template: str = (
        "Dựa vào các đoạn văn bản sau:\n{context}\n\nTrả lời: {question}"
    )
    reranker_instruction: str = (
        "Cho một câu hỏi về kinh tế, tài chính, đánh giá mức độ liên quan của đoạn văn bản với câu hỏi"
    )
    no_context_message: str = (
        "Không tìm thấy dữ liệu phù hợp trong tài liệu nội bộ hoặc nguồn web hiện có."
    )
    guard_error_message: str = (
        "Xin lỗi, tôi không thể xử lý yêu cầu của bạn do yêu cầu đã vi phạm chính sách của chúng tôi."
    )
    apology_message: str = (
        "Xin lỗi, tôi không thể trả lời câu hỏi này theo tài liệu hiện tại."
    )


class ObservabilityConfig(BaseModel):
    log_level: str
    langsmith_api_key: str | None = None
    langsmith_project: str
    tavily_api_key: str | None = None


class Settings(BaseSettings):
    llm: LLMConfig
    services: ServicesConfig
    rag: RAGConfig
    prompts: PromptsConfig = PromptsConfig()
    observability: ObservabilityConfig

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        env_nested_delimiter="__",
        extra="ignore",
    )


@lru_cache()
def get_settings() -> Settings:
    return Settings()
