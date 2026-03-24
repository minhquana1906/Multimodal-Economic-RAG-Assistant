from __future__ import annotations

import time
import uuid
from typing import Literal

from pydantic import BaseModel, Field


class TextContentPart(BaseModel):
    type: Literal["text"] = "text"
    text: str


MessageContent = str | list[TextContentPart]


class Message(BaseModel):
    role: Literal["system", "user", "assistant"]
    content: MessageContent

    def text_content(self) -> str:
        if isinstance(self.content, str):
            return self.content
        return "".join(part.text for part in self.content if part.type == "text")


class ChatRequest(BaseModel):
    model: str
    messages: list[Message] = Field(min_length=1)
    temperature: float = Field(default=0.7, ge=0.0, le=2.0)
    max_tokens: int | None = Field(default=None, gt=0)
    stream: bool = False


class AssistantMessage(BaseModel):
    role: Literal["assistant"] = "assistant"
    content: str


class ChatDelta(BaseModel):
    role: str | None = None
    content: str = ""


class ChatCompletionChoice(BaseModel):
    message: AssistantMessage
    index: int = 0
    finish_reason: str | None = "stop"


class ChatStreamChoice(BaseModel):
    delta: ChatDelta
    index: int = 0
    finish_reason: str | None = None


class ChatResponse(BaseModel):
    id: str = Field(default_factory=lambda: f"chatcmpl-{uuid.uuid4().hex[:8]}")
    object: str = "chat.completion"
    created: int = Field(default_factory=lambda: int(time.time()))
    model: str
    choices: list[ChatCompletionChoice]


class ChatStreamChunk(BaseModel):
    id: str = Field(default_factory=lambda: f"chatcmpl-{uuid.uuid4().hex[:8]}")
    object: str = "chat.completion.chunk"
    created: int = Field(default_factory=lambda: int(time.time()))
    model: str
    choices: list[ChatStreamChoice]


class ChunkContext(BaseModel):
    context_id: str = ""
    text: str
    source_type: str = ""
    retrieval_stage: str = ""
    original_rank: int = -1
    collection_name: str = ""
    doc_type: str = ""
    chunk_type: str = ""
    modality: str = "text"
    source_quality: str = ""
    source: str = ""
    title: str = ""
    url: str = ""
    score: float = 0.0
    image_path: str = ""
    structured_data: dict = Field(default_factory=dict)
