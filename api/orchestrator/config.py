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
    inference_url: str
    inference_timeout: float
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
    intent_system_prompt: str = (
        "Bạn là bộ định tuyến route cho trợ lý kinh tế - tài chính.\n"
        "Chỉ trả về JSON hợp lệ với hai khóa: route và resolved_query.\n"
        'route phải là "direct" hoặc "rag".\n'
        "Nếu không chắc, chọn rag."
    )
    intent_user_template: str = (
        "Phân tích các tin nhắn sau và trả về JSON theo đúng schema đã yêu cầu.\n\n"
        "{messages}"
    )
    direct_system_prompt: str = (
        "Viết lại hoặc trả lời trực tiếp bằng tiếng Việt tự nhiên.\n"
        "Không dùng citations. Không bịa dữ kiện cần tra cứu."
    )
    rag_system_prompt: str = (
        "Bạn là trợ lý AI về kinh tế và tài chính với giọng điệu ấm áp, nhẹ nhàng, điềm tĩnh và thiên hướng học thuật.\n"
        "Mục tiêu của bạn là giải thích rõ ràng cho người dùng phổ thông bằng tiếng Việt.\n"
        "Phạm vi có thể bao phủ bối cảnh toàn cầu, nhưng luôn ưu tiên Việt Nam khi thông tin có liên quan.\n"
        "Bạn chỉ được khẳng định điều có cơ sở từ nguồn đã cung cấp và phải nói rõ giới hạn khi bằng chứng chưa đủ."
    )
    rag_user_template: str = (
        "Ngữ cảnh hội thoại:\n{conversation_context}\n\n"
        "Nguồn đã gán ID:\n{context}\n\n"
        "Câu hỏi đã làm rõ:\n{question}"
    )
    no_context_message: str = (
        "Không tìm thấy dữ liệu phù hợp trong tài liệu nội bộ hoặc nguồn web hiện có."
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
