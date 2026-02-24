"""Request/response for SDXL production API. Style values must match Style enum."""
from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, Field

# Must match Style enum values exactly (no fallback)
StyleKind = Literal[
    "anime",
    "realistic",
    "watercolor",
    "cyberpunk",
    "oil_painting",
    "sketch",
    "cinematic",
    "fantasy_art",
    "pixel_art",
    "3d_render",
]


class GenerateRequest(BaseModel):
    prompt: str = Field(..., min_length=1, max_length=2000)
    style: StyleKind = Field(..., description="Must be one of Style enum values")
    negative_prompt: str = Field(default="", max_length=2000)
    steps: int = Field(default=28, ge=1, le=100)
    cfg: float = Field(default=7.0, ge=1.0, le=20.0)
    seed: int | None = Field(default=None, ge=0)
    width: int = Field(default=768, ge=256, le=1024)
    height: int = Field(default=768, ge=256, le=1024)


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
