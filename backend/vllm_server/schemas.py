"""
OpenAI-compatible request/response schemas. Strict validation, no 'any'.
"""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, ConfigDict, Field


class ChatMessage(BaseModel):
    role: Literal["system", "user", "assistant"]
    content: str


class ChatCompletionRequest(BaseModel):
    model: str = Field(default="gpt-oss-20b", description="Model name (may be ignored by backend).")
    messages: list[ChatMessage] = Field(..., min_length=1)
    stream: bool = Field(default=False)
    max_tokens: int = Field(default=256, ge=1, le=4096)
    temperature: float = Field(default=0.7, ge=0.0, le=2.0)
    top_p: float = Field(default=1.0, ge=0.0, le=1.0)
    stop: list[str] | None = Field(default=None)
    request_id: str | None = Field(default=None)


class ChatCompletionChoice(BaseModel):
    index: int
    message: ChatMessage
    finish_reason: Literal["stop", "length", "content_filter", "null"] = "stop"


class ChatCompletionResponse(BaseModel):
    model_config = ConfigDict(extra="ignore")
    id: str
    object: Literal["chat.completion"] = "chat.completion"
    created: int = 0
    model: str
    choices: list[ChatCompletionChoice]
    usage: dict[str, int] | None = None


class StreamChoiceDelta(BaseModel):
    role: Literal["assistant"] | None = None
    content: str | None = None


class StreamChoice(BaseModel):
    index: int
    delta: StreamChoiceDelta
    finish_reason: Literal["stop", "length", "content_filter", "null"] | None = None


class ChatCompletionChunk(BaseModel):
    id: str
    object: Literal["chat.completion.chunk"] = "chat.completion.chunk"
    created: int = 0
    model: str
    choices: list[StreamChoice]
