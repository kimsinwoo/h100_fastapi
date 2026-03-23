"""
DTOs for dance generation (pose → frames → video).

Internal and API-facing models use strict typing (no Any in public fields).
"""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, Field


class CharacterPreparationResult(BaseModel):
    """Result of validating / preparing the uploaded character image."""

    ok: bool = Field(..., description="Whether the image is valid for generation")
    width: int = Field(..., ge=1)
    height: int = Field(..., ge=1)
    format: str = Field(..., description="Detected image format, e.g. JPEG, PNG")
    comfy_upload_name: str | None = Field(
        default=None,
        description="ComfyUI input image name after upload (if ComfyUI path used)",
    )


class PoseExtractionResult(BaseModel):
    """Pose sequence extraction outcome."""

    cache_path: str = Field(..., description="Path to cached MotionSequence JSON")
    frame_count: int = Field(..., ge=0)
    fps: float = Field(..., gt=0)
    source_video_path: str = Field(..., description="Reference dance video path used")


class FrameBatchSpec(BaseModel):
    """One batch of frame indices to render through ComfyUI."""

    start_index: int = Field(..., ge=0)
    end_index: int = Field(..., ge=0)
    frame_indices: list[int] = Field(default_factory=list)


class DanceGenerationJobConfig(BaseModel):
    """Runtime configuration for a single dance generation job."""

    motion_id: str = Field(..., min_length=1)
    character: Literal["dog", "cat"] = "dog"
    pipeline: Literal["ltx", "pose_sdxl"] = Field(
        default="ltx",
        description="ltx: LTX + reference video (ComfyUI/diffusers). pose_sdxl: pose extract → ComfyUI frames → ffmpeg.",
    )
    seed: int | None = Field(default=None, description="Global seed for reproducibility")
    batch_size: int = Field(default=4, ge=1, le=32)
    max_frames: int = Field(default=49, ge=8, le=241, description="Cap frames for SDXL path")


class DanceGenerationInternalState(BaseModel):
    """Optional debug state returned to logs (not exposed on public API by default)."""

    pipeline: Literal["ltx_reference_video", "comfyui_sdxl_pose_frames"] = "ltx_reference_video"
    pose_cache_path: str | None = None
    frame_png_count: int | None = None
