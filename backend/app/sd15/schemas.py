"""Request/response and error schemas for SD 1.5 API."""
from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, Field


StyleKind = Literal["anime", "realistic"]


class GenerateRequest(BaseModel):
    prompt: str = Field(..., min_length=1, max_length=2000)
    style: StyleKind = Field(default="realistic")
    strength: float | None = Field(default=None, ge=0.0, le=1.0)
    steps: int | None = Field(default=None, ge=1, le=100)
    cfg: float | None = Field(default=None, ge=1.0, le=20.0)
    image_base64: str | None = Field(default=None)
    upscale: bool | None = Field(default=None)


class GenerateResponse(BaseModel):
    image_base64: str = Field(...)
    seed: int = Field(...)
    style: str = Field(...)


class ErrorDetail(BaseModel):
    code: str = Field(...)
    message: str = Field(...)


class ErrorResponse(BaseModel):
    error: ErrorDetail = Field(...)


def error_response(code: str, message: str) -> ErrorResponse:
    return ErrorResponse(error=ErrorDetail(code=code, message=message))
