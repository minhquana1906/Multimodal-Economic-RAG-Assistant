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
    query_rewrite_prompt: str = (
        "Bạn là bộ chuẩn hóa truy vấn cho hệ thống RAG.\n"
        "Viết lại thành đúng 1 câu hỏi hoàn chỉnh bằng tiếng Việt.\n"
        "Giữ nguyên ý định người dùng, sửa chính tả nếu cần, làm rõ đại từ tham chiếu bằng ngữ cảnh liên quan.\n"
        "Không thêm thông tin mới ngoài hội thoại.\n"
        "Chỉ trả về đúng câu hỏi đã được viết lại, không giải thích thêm."
    )
    route_classifier_prompt: str = (
        "Bạn là bộ phân loại route hội thoại cho trợ lý đa năng.\n"
        "Chỉ trả về đúng một nhãn: rag hoặc general_chat.\n"
        "- Chọn rag nếu truy vấn liên quan đến kinh tế, tài chính, số liệu, doanh nghiệp, chính sách, thị trường, phân tích, so sánh, dự báo, hoặc cần grounding theo ngữ cảnh.\n"
        "- Chọn general_chat nếu là xã giao, meta-chat, hỏi khả năng trợ lý, hoặc yêu cầu broad assistant ngoài domain như viết lại câu, chỉnh wording, hỗ trợ giao tiếp chung.\n"
        "- Nếu không chắc, chọn rag.\n\n"
        "Truy vấn đã làm rõ:\n{resolved_query}\n\n"
        "Ngữ cảnh hội thoại:\n{conversation_context}"
    )
    rag_system_prompt: str = (
        "Bạn là trợ lý AI về kinh tế và tài chính với giọng điệu ấm áp, nhẹ nhàng, điềm tĩnh và thiên hướng học thuật.\n"
        "Mục tiêu của bạn là giải thích rõ ràng cho người dùng phổ thông bằng tiếng Việt.\n"
        "Phạm vi có thể bao phủ bối cảnh toàn cầu, nhưng luôn ưu tiên Việt Nam khi thông tin có liên quan.\n"
        "Bạn chỉ được khẳng định điều có cơ sở từ nguồn đã cung cấp và phải nói rõ giới hạn khi bằng chứng chưa đủ."
    )
    rag_user_template: str = (
        "{conversation_context}\n\n"
        "{response_contract}\n\n"
        "Quy tắc dùng nguồn:\n"
        "- Ưu tiên tài liệu nội bộ khi đã đủ thông tin.\n"
        "- Nếu tài liệu nội bộ chưa đủ, dùng thêm nguồn web được cung cấp.\n"
        "- Khi nguồn nội bộ và web khác nhau, phải đối chiếu điểm giống và điểm khác trước khi kết luận.\n"
        "- Không khẳng định các ý không có trong nguồn được cung cấp.\n"
        "- Nếu nguồn chưa đủ, nêu rõ giới hạn dữ liệu một cách bình tĩnh, chính xác.\n\n"
        "Nguồn được cung cấp:\n{context}\n\n"
        "Truy vấn đã làm rõ:\n{question}"
    )
    rag_text_response_contract: str = (
        "Yêu cầu định dạng câu trả lời:\n"
        "- Trả lời bằng tiếng Việt.\n"
        "- Trình bày bằng markdown rõ ràng, tự nhiên, dễ đọc.\n"
        "- Mặc định chia câu trả lời thành 2-4 phần chính với header `##`; tiêu đề do bạn tự đặt theo nội dung thay vì dùng mẫu cố định.\n"
        "- Ngăn cách các phần bằng một dòng `---` để bố cục rõ ràng hơn.\n"
        "- Trong từng phần, ưu tiên văn xuôi tự nhiên và giải thích chi tiết, rõ ràng hơn một chút so với trả lời quá ngắn.\n"
        "- Ưu tiên dùng gạch đầu dòng khi đang liệt kê ý, điều kiện, tác động, hoặc đối chiếu nguồn; không lạm dụng bullet point nếu đoạn văn sẽ tự nhiên hơn.\n"
        "- Khi cần đối chiếu nguồn hoặc nêu giới hạn dữ liệu, hãy dành một phần riêng với header phù hợp do bạn tự đặt.\n"
        "- Văn phong học thuật nhưng dễ hiểu với người dùng phổ thông.\n"
        "- Không lan man, không khẳng định quá mức, không tạo mục rỗng không cần thiết."
    )
    rag_audio_response_contract: str = (
        "Yêu cầu định dạng câu trả lời:\n"
        "- Trả lời bằng tiếng Việt trong một đoạn ngắn, tự nhiên như văn nói.\n"
        "- Giữ văn nói tự nhiên, ấm áp, nhẹ nhàng, đi thẳng vào ý chính.\n"
        "- Giữ giọng điệu ấm áp, nhẹ nhàng, đi thẳng vào ý chính.\n"
        "- Chỉ nên dài 1-3 câu ngắn.\n"
        "- Không dùng markdown, bullet point, emoji, citation footer, hay ký tự đặc biệt không cần thiết."
    )
    general_chat_system_prompt: str = (
        "Bạn là trợ lý AI nói tiếng Việt với phong cách ấm áp, nhẹ nhàng, tự nhiên và hữu ích.\n"
        "Bạn có thể hỗ trợ xã giao, meta-chat, giải thích chung, và các yêu cầu broad assistant ngoài domain.\n"
        "Hãy trả lời rõ ràng, thân thiện, không màu mè, không giả học thuật với các câu xã giao đơn giản."
    )
    general_chat_user_template: str = (
        "{conversation_context}\n\n"
        "{response_contract}\n\n"
        "Người dùng đang hỏi:\n{question}"
    )
    general_chat_text_response_contract: str = (
        "Yêu cầu định dạng câu trả lời:\n"
        "- Trả lời bằng tiếng Việt tự nhiên như hội thoại.\n"
        "- Ưu tiên 1-2 đoạn ngắn, chỉ dùng bullet nếu người dùng yêu cầu liệt kê hoặc nội dung thực sự cần liệt kê.\n"
        "- Giữ giọng ấm áp, nhẹ nhàng, rõ ràng, không sáo rỗng."
    )
    general_chat_audio_response_contract: str = (
        "Yêu cầu định dạng câu trả lời:\n"
        "- Trả lời bằng tiếng Việt tự nhiên như văn nói.\n"
        "- Chỉ nên dài 1-3 câu ngắn.\n"
        "- Không dùng emoji, markdown, bullet point, hay ký tự đặc biệt."
    )
    general_chat_live_facts_message: str = (
        "Để trả lời chính xác câu này, mình cần tra cứu nguồn hoặc công cụ có dữ liệu cập nhật theo thời gian thực."
    )
    general_chat_retry_prompt: str = (
        "{original_prompt}\n\n"
        "Bản nháp trước đó đã bị bộ lọc an toàn đánh dấu là không phù hợp.\n"
        "Safety: {safe_label}\n"
        "Categories: {categories}\n"
        "Refusal: {refusal}\n\n"
        "Bản nháp không an toàn:\n{unsafe_answer}\n\n"
        "Hãy tạo lại câu trả lời bằng tiếng Việt, ngắn gọn, lịch sự, an toàn và không lặp lại nội dung bị chặn. "
        "Nếu không thể trả lời an toàn, hãy từ chối ngắn gọn."
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

    @property
    def system_prompt(self) -> str:
        return self.rag_system_prompt

    @property
    def user_template(self) -> str:
        return self.rag_user_template


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
