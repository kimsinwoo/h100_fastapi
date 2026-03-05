"""
Request/response schemas for image generation API.
"""

from __future__ import annotations

from pydantic import BaseModel, Field


class GenerateResponse(BaseModel):
    original_url: str
    generated_url: str
    processing_time: float = Field(..., description="Seconds taken for inference")
    # 멀티 Pod/다른 호스트에서 GET /static/... 404 방지: 생성 직후 바이너리를 base64로 함께 반환
    generated_image_base64: str | None = Field(default=None, description="PNG bytes as base64; use when GET generated_url returns 404")


class GenerateVideoResponse(BaseModel):
    """LTX-2 이미지→동영상 생성 응답."""
    video_url: str = Field(..., description="생성된 동영상 URL (e.g. /static/generated/xxx.mp4)")
    processing_time: float = Field(..., description="초 단위 소요 시간")
    video_base64: str | None = Field(default=None, description="mp4 base64 (선택, 멀티 Pod 대응)")


class ErrorDetail(BaseModel):
    detail: str
    code: str | None = None


# ---------- AC Villager Reconstruction Pipeline ----------


class ACBiologicalAnalysis(BaseModel):
    """Stage 1 output: biological traits extracted from uploaded image (structure only, no rendering)."""
    species: str = Field(..., description="e.g. cat, dog, rabbit, hamster, bird")
    main_fur_color: str = Field(..., description="Primary fur color")
    secondary_fur_color: str = Field(default="none", description="Secondary fur or none")
    eye_color: str = Field(..., description="Eye color")
    markings: str = Field(default="none", description="Major markings only, large patches")
    ear_type: str = Field(..., description="Ear shape description")
    tail_type: str = Field(default="simplified", description="Tail shape description")


class ACReconstructRequest(BaseModel):
    """Stage 2 input: biological data only (no image). Used for text-to-image-only villager generation."""
    species: str = Field(..., description="cat, dog, rabbit, hamster, bird, other")
    main_fur_color: str = Field(default="cream")
    secondary_fur_color: str = Field(default="none")
    eye_color: str = Field(default="amber")
    markings: str = Field(default="none")
    ear_type: str | None = Field(default=None, description="Optional; inferred from species if omitted")
    tail_type: str | None = Field(default=None, description="Optional; simplified if omitted")
    seed: int | None = Field(default=None)
