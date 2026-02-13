"""
Request/response schemas for image generation API.
"""

from __future__ import annotations

from pydantic import BaseModel, Field


class GenerateResponse(BaseModel):
    original_url: str
    generated_url: str
    processing_time: float = Field(..., description="Seconds taken for inference")


class ErrorDetail(BaseModel):
    detail: str
    code: str | None = None
