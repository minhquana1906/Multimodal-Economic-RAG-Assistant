from __future__ import annotations

from functools import lru_cache
from typing import Literal

from pydantic import AliasChoices, BaseModel, Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class LLMConfig(BaseModel):
    url: str
    model: str
    temperature: float
    max_tokens: int
    timeout: float
    api_key: str = ""
    enable_vision: bool = True
    max_image_pixels: int = 1_048_576
    max_image_bytes: int = 4_000_000
    image_detail: Literal["auto", "low", "high"] = "auto"
    max_images_per_turn: int = 4


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
        "direct: câu hỏi đơn giản, chào hỏi, viết lại, diễn giải, không cần tra cứu tài liệu.\n"
        "rag: câu hỏi về kinh tế, tài chính, số liệu, phân tích, cần tra cứu tài liệu hoặc nguồn tham chiếu.\n"
        "Nếu không chắc, chọn rag."
    )
    intent_user_template: str = (
        "Phân tích các tin nhắn sau và trả về JSON theo đúng schema đã yêu cầu.\n\n"
        "{messages}"
    )
    direct_system_prompt: str = (
        "Bạn là trợ lý AI về kinh tế và tài chính với giọng điệu ấm áp, nhẹ nhàng, điềm tĩnh.\n"
        "Mục tiêu là trả lời hoặc viết lại trực tiếp bằng ngôn ngữ giống với ngôn ngữ câu hỏi sử dụng 1 cách tự nhiên, sáng sủa, có cấu trúc rõ ràng và dễ quét mắt.\n"
    )
    direct_response_contract: str = (
        "Yêu cầu định dạng câu trả lời:\n"
        "- Trả lời bằng ngôn ngữ giống với ngôn ngữ câu hỏi sử dụng 1 cách tự nhiên, giàu thông tin nhưng không lan man, sử dụng markdown.\n"
        "- Mặc định chia câu trả lời thành 2-4 phần chính với header `##`; tự đặt tiêu đề sát nội dung.\n"
        "- Nếu câu trả lời đủ dài để tách phần, ngăn cách các phần bằng một dòng `---`.\n"
        "- Ưu tiên dùng gạch đầu dòng khi liệt kê, dùng bảng khi so sánh, dùng đoạn văn khi giải thích.\n"
        "- Tránh trả lời thành một khối văn bản dài; nên có phần kết ngắn hoặc gợi ý tiếp theo khi phù hợp."
    )
    direct_user_template: str = (
        "{response_contract}\n\n"
        "Hội thoại gần đây:\n{conversation}\n\n"
        "Yêu cầu hiện tại đã làm rõ:\n{question}"
    )
    rag_system_prompt: str = (
        "Bạn là trợ lý AI về kinh tế và tài chính với giọng điệu ấm áp, nhẹ nhàng, điềm tĩnh và thiên hướng học thuật.\n"
        "Mục tiêu của bạn là giải thích rõ ràng cho người dùng phổ thông bằng ngôn ngữ giống với ngôn ngữ câu hỏi sử dụng.\n"
        "Bạn chỉ được khẳng định điều có cơ sở từ nguồn đã cung cấp và phải nói rõ giới hạn khi bằng chứng chưa đủ.\n"
        "Khi trả lời, bắt buộc trích dẫn inline bằng [S1], [S2], ... ngay sau mỗi khẳng định có căn cứ từ nguồn."
    )
    rag_text_response_contract: str = (
        "Yêu cầu định dạng câu trả lời:\n"
        "- Trả lời bằng markdown rõ ràng, tự nhiên, dễ đọc.\n"
        "- Luôn chia câu trả lời thành 2-4 phần chính, mỗi phần có header `##`; tự đặt tiêu đề phù hợp nội dung.\n"
        "- Ngăn cách các phần bằng một dòng `---` để bố cục rõ ràng.\n"
        "- Ưu tiên dùng gạch đầu dòng khi liệt kê, dùng bảng khi so sánh, dùng đoạn văn khi giải thích.\n"
        "- Giọng điệu ấm áp, súc tích; không lan man, không khẳng định quá mức, không tạo mục rỗng."
    )
    rag_user_template: str = (
        "{response_contract}\n\n"
        "Nguồn đã gán ID:\n{context}\n\n"
        "Câu hỏi đã làm rõ:\n{question}"
    )
    no_context_message: str = (
        "Không tìm thấy dữ liệu phù hợp trong tài liệu nội bộ hoặc nguồn web hiện có."
    )
    image_caption_system_prompt: str = (
        "Bạn là mô-đun trích xuất ý nghĩa hình ảnh cho hệ thống RAG kinh tế - tài chính.\n"
        "Chỉ trả về JSON hợp lệ duy nhất với 2 khóa: caption và rag_query.\n"
        "caption: mô tả ngắn gọn 1-2 câu về nội dung ảnh (số liệu, biểu đồ, bảng, văn bản).\n"
        "rag_query: 1 truy vấn tiếng Việt súc tích để tra cứu thông tin kinh tế liên quan trong cơ sở dữ liệu."
    )
    image_caption_user_template: str = (
        "Yêu cầu của người dùng: {user_text}\n\n"
        "Hãy phân tích ảnh kèm theo và trả về JSON:\n"
        '{{\"caption\": \"mô tả 1-2 câu\", \"rag_query\": \"truy vấn tiếng Việt\"}}'
    )
    intent_user_template_with_image: str = (
        "Phân tích các tin nhắn sau và trả về JSON theo đúng schema đã yêu cầu.\n\n"
        "Ảnh đính kèm: {image_caption}\n\n"
        "{messages}"
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
