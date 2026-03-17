"""
Pydantic schemas for Reels dance LoRA API.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional

from pydantic import BaseModel, Field


# ──────────────────────────────────────────────────────────────────────────────
# Registry
# ──────────────────────────────────────────────────────────────────────────────

class LoRAEntryResponse(BaseModel):
    category: str
    display_name: str
    is_loaded: bool
    lora_rank: int
    base_model: str
    weights_path: str


class RegistryInfoResponse(BaseModel):
    total: int
    categories: List[str]
    loaded: List[str]
    entries: List[LoRAEntryResponse]


# ──────────────────────────────────────────────────────────────────────────────
# Training
# ──────────────────────────────────────────────────────────────────────────────

class TrainingRequest(BaseModel):
    category: str = Field(
        ...,
        description="Dance category name, e.g. 'tiktok_shuffle'",
        examples=["tiktok_shuffle"],
    )
    video_dir: str = Field(
        ...,
        description="Absolute or backend-relative path to directory of dance videos",
    )
    num_train_steps: int = Field(default=5000, ge=100, le=50000)
    lora_rank: int = Field(default=32, ge=4, le=128)
    lora_alpha: int = Field(default=32, ge=4, le=128)
    learning_rate: float = Field(default=1e-4, ge=1e-6, le=1e-2)
    batch_size: int = Field(default=2, ge=1, le=16)
    clip_frames: int = Field(default=16, ge=8, le=32)
    image_size: int = Field(default=512, ge=256, le=768)
    gradient_accumulation_steps: int = Field(default=4, ge=1, le=32)
    save_every: int = Field(default=500, ge=50, le=5000)
    enable_torch_compile: bool = Field(default=False)


class TrainingJobResponse(BaseModel):
    job_id: str
    category: str
    status: str   # "queued" | "running" | "completed" | "failed"
    message: str


class TrainingStatusResponse(BaseModel):
    job_id: str
    category: str
    status: str
    step: Optional[int] = None
    total_steps: Optional[int] = None
    loss: Optional[float] = None
    output_path: Optional[str] = None
    error: Optional[str] = None
    elapsed_seconds: Optional[float] = None


# ──────────────────────────────────────────────────────────────────────────────
# Generation
# ──────────────────────────────────────────────────────────────────────────────

class ReelsGenerateJobResponse(BaseModel):
    job_id: str
    category: str
    status: str   # "processing" | "completed" | "failed"


class ReelsGenerateStatusResponse(BaseModel):
    job_id: str
    status: str
    category: Optional[str] = None
    character: Optional[str] = None
    video_url: Optional[str] = None
    processing_time: Optional[float] = None
    error: Optional[str] = None
    mode: Optional[str] = None   # "animatediff" | "ltx2"
