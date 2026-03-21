from functools import lru_cache

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    # Service URLs
    embedding_url: str = "http://embedding:8001"
    reranker_url: str = "http://reranker:8002"
    guard_url: str = "http://guard:8003"
    qdrant_url: str = "http://qdrant:6333"
    llm_url: str = "http://localhost:8004"

    # Qdrant
    qdrant_collection: str = "econ_vn_news"

    # RAG params
    retrieval_top_k: int = 20
    rerank_top_n: int = 5
    fallback_min_chunks: int = 3
    fallback_score_threshold: float = 0.5

    # Timeouts
    guard_timeout: float = 10.0
    llm_timeout: float = 60.0
    embedding_timeout: float = 15.0
    reranker_timeout: float = 15.0

    # Messages (Vietnamese)
    no_context_message: str = "Xin lỗi, tôi không tìm thấy thông tin liên quan."
    guard_error_message: str = "Xin lỗi, yêu cầu của bạn không thể xử lý."

    # LangSmith
    langsmith_api_key: str | None = None
    langsmith_project: str = "multimodal-rag"

    model_config = SettingsConfigDict(env_file=".env", env_file_encoding="utf-8")


@lru_cache()
def get_settings() -> Settings:
    return Settings()
