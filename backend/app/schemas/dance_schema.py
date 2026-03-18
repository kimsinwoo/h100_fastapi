"""
Dance / Motion Transfer API schemas.
Pose keypoints follow OpenPose-style body skeleton (compatible with MediaPipe mapping).
"""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, Field


class Keypoint(BaseModel):
    """Single body joint in image coordinates."""

    joint: str = Field(..., description="Joint name (e.g. nose, neck, left_shoulder)")
    x: float = Field(..., description="Normalized or pixel x")
    y: float = Field(..., description="Normalized or pixel y")


class FramePose(BaseModel):
    """Pose for one frame: frame index, timestamp, keypoints."""

    frame: int = Field(..., ge=0, description="Frame index (0-based)")
    timestamp: float = Field(..., ge=0, description="Time in seconds")
    keypoints: list[Keypoint] = Field(default_factory=list, description="Body skeleton keypoints")


class MotionSequence(BaseModel):
    """Full pose sequence for a video (extraction output)."""

    fps: float = Field(..., gt=0, description="Video FPS")
    width: int = Field(..., ge=1)
    height: int = Field(..., ge=1)
    frames: list[FramePose] = Field(default_factory=list, description="Pose per frame")


class DanceGenerateRequest(BaseModel):
    """Request for dance video generation (body only; file comes via Form)."""

    motion_id: str = Field(..., min_length=1, max_length=64, description="e.g. rat_dance")
    character: Literal["dog", "cat"] = Field(default="dog", description="Character to perform the dance")


class DanceGenerateResponse(BaseModel):
    """Response after dance video generation."""

    video_url: str = Field(..., description="URL or path to generated video")
    processing_time: float = Field(..., ge=0, description="Processing time in seconds")
    motion_id: str = Field(..., description="Echo of motion_id")
    character: str = Field(..., description="Echo of character")


class DanceJobResponse(BaseModel):
    """Immediate response for async dance job — returns job_id for polling."""

    job_id: str
    status: str = "processing"


class DanceJobStatusResponse(BaseModel):
    """GET /dance/status/{job_id} polling response."""

    job_id: str
    status: str                          # "processing" | "completed" | "failed"
    video_url: str | None = None
    processing_time: float | None = None
    motion_id: str | None = None
    character: str | None = None
    error: str | None = None


# ----- 댄스 라이브러리 (사전 등록 영상 목록) -----


class DanceVideoInfo(BaseModel):
    """사전 등록 댄스 영상 한 건 (GET /api/dance/list 응답용)."""

    id: str = Field(..., description="dance_video_id (파일명 기반 slug)")
    display_name: str = Field(..., description="표시 이름")
    filename: str = Field(..., description="파일명")
    duration_seconds: float = Field(..., ge=0)
    fps: float = Field(..., ge=0)
    width: int = Field(..., ge=0)
    height: int = Field(..., ge=0)
    frame_count: int = Field(..., ge=0)
    file_size_mb: float = Field(..., ge=0)


class DanceListResponse(BaseModel):
    """GET /api/dance/list 응답."""

    total: int = Field(..., description="등록된 댄스 영상 개수")
    dances: list[DanceVideoInfo] = Field(default_factory=list)


class DanceRefreshResponse(BaseModel):
    """POST /api/dance/refresh 응답."""

    message: str = Field(..., description="예: 스캔 완료")
    added: list[str] = Field(default_factory=list, description="새로 추가된 파일명")
    total: int = Field(..., ge=0, description="스캔 후 전체 개수")
