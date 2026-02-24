"""Request/response and structured errors for SDXL production API."""
from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, Field

StyleKind = Literal[
    "anime",
    "realistic",
    "cinematic",
    "3d_render",
    "cyberpunk",
    "fantasy",
    "fantasy_art",
    "watercolor",
    "oil_painting",
    "sketch",
    "pixel_art",
]


class GenerateRequest(BaseModel):
    prompt: str = Field(..., min_length=1, max_length=2000)
    style: StyleKind = Field(default="realistic")
    strength: float | None = Field(default=None, ge=0.0, le=1.0)
    steps: int | None = Field(default=None, ge=1, le=100)
    cfg: float | None = Field(default=None, ge=1.0, le=20.0)
    image_base64: str | None = Field(default=None)
    width: int | None = Field(default=None, ge=256, le=1024)
    height: int | None = Field(default=None, ge=256, le=1024)


class GenerateResponse(BaseModel):
    image_base64: str = Field(...)
    seed: int = Field(...)
    style: str = Field(...)
    width: int = Field(...)
    height: int = Field(...)


class ErrorDetail(BaseModel):
    code: str = Field(...)
    message: str = Field(...)


class ErrorResponse(BaseModel):
    error: ErrorDetail = Field(...)


def error_response(code: str, message: str) -> ErrorResponse:
    return ErrorResponse(error=ErrorDetail(code=code, message=message))
