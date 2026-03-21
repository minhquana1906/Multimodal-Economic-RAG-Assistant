from __future__ import annotations

from pydantic import BaseModel


class Message(BaseModel):
    role: str
    content: str


class ChatRequest(BaseModel):
    model: str
    messages: list[Message]
    temperature: float = 0.7
    max_tokens: int | None = None
    stream: bool = False


class ChatDelta(BaseModel):
    role: str | None = None
    content: str = ""


class ChatChoice(BaseModel):
    delta: ChatDelta
    index: int = 0
    finish_reason: str | None = None


class ChatResponse(BaseModel):
    model: str
    choices: list[ChatChoice]


class ChunkContext(BaseModel):
    text: str
    source: str = ""
    title: str = ""
    score: float = 0.0
