"""
Pydantic v2 request/response schemas. No deprecated patterns.
"""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, Field


class GenerateRequest(BaseModel):
    prompt: str = Field(..., min_length=1, max_length=2000)
    negative_prompt: str = Field(default="", max_length=2000)
    style: str = Field(default="realistic", max_length=64)
    strength: float = Field(default=0.75, ge=0.0, le=1.0)
    steps: int = Field(default=30, ge=1, le=100)
    cfg: float = Field(default=7.5, ge=1.0, le=20.0)
    seed: int | None = Field(default=None, ge=0, le=2**32 - 1)
    width: int = Field(default=1024, ge=512, le=2048)
    height: int = Field(default=1024, ge=512, le=2048)
    lora_path: str | None = Field(default=None, max_length=512)
    lora_scale: float = Field(default=0.85, ge=0.0, le=2.0)


class GenerateResponse(BaseModel):
    image_url: str
    processing_time_seconds: float


class TrainLoraRequest(BaseModel):
    dataset_path: str = Field(..., min_length=1, max_length=1024)
    output_path: str = Field(..., min_length=1, max_length=1024)
    rank: int = Field(default=4, ge=1, le=64)
    learning_rate: float = Field(default=1e-4, ge=1e-6, le=1e-2)
    steps: int = Field(default=500, ge=10, le=10000)
    batch_size: int = Field(default=1, ge=1, le=8)
    resolution: int = Field(default=1024, ge=512, le=1024)


class TrainLoraResponse(BaseModel):
    status: Literal["started", "completed", "failed", "error", "timeout"]
    output_path: str
    steps: int
    message: str | None = None
    error: str | None = None


class HealthResponse(BaseModel):
    status: Literal["ok", "degraded"]
    gpu_available: bool
    models_loaded: list[str] = Field(default_factory=list)
