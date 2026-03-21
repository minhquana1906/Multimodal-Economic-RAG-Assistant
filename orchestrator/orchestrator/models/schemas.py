from __future__ import annotations

import time
import uuid
from typing import Literal

from pydantic import BaseModel, Field


class Message(BaseModel):
    role: Literal["system", "user", "assistant"]
    content: str


class ChatRequest(BaseModel):
    model: str
    messages: list[Message] = Field(min_length=1)
    temperature: float = Field(default=0.7, ge=0.0, le=2.0)
    max_tokens: int | None = Field(default=None, gt=0)
    stream: bool = False


class ChatDelta(BaseModel):
    role: str | None = None
    content: str = ""


class ChatChoice(BaseModel):
    delta: ChatDelta
    index: int = 0
    finish_reason: str | None = None


class ChatResponse(BaseModel):
    id: str = Field(default_factory=lambda: f"chatcmpl-{uuid.uuid4().hex[:8]}")
    object: str = "chat.completion"
    created: int = Field(default_factory=lambda: int(time.time()))
    model: str
    choices: list[ChatChoice]


class ChatStreamChunk(BaseModel):
    id: str = Field(default_factory=lambda: f"chatcmpl-{uuid.uuid4().hex[:8]}")
    object: str = "chat.completion.chunk"
    created: int = Field(default_factory=lambda: int(time.time()))
    model: str
    choices: list[ChatChoice]


class ChunkContext(BaseModel):
    text: str
    source: str = ""
    title: str = ""
    score: float = 0.0
