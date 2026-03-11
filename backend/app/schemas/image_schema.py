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


class VideoJobResponse(BaseModel):
    """동영상 생성 비동기: POST 즉시 반환."""
    job_id: str = Field(..., description="폴링용 job_id. GET /api/video/status/{job_id} 로 결과 조회")


class VideoJobStatusResponse(BaseModel):
    """GET /api/video/status/{job_id} 응답."""
    status: str = Field(..., description="processing | completed | failed")
    video_url: str | None = None
    processing_time: float | None = None
    video_base64: str | None = None
    error: str | None = None


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


# ---------- Image Analysis (structured visual attributes, JSON only) ----------


class AnimalInfo(BaseModel):
    species: str = Field(..., description="cat / dog / other")
    breed: str | None = Field(default=None, description="if recognizable")
    fur_main_color: str = Field(default="unknown")
    fur_secondary_color: str = Field(default="unknown")
    major_markings: str = Field(default="unknown")


class ClothingDetection(BaseModel):
    is_wearing_clothes: bool = Field(default=False)
    clothing_type: str = Field(default="unknown", description="shirt / hoodie / dress / sweater / costume / etc")
    clothing_color: str = Field(default="unknown")
    clothing_pattern: str = Field(default="unknown")
    sleeve_length: str = Field(default="unknown")
    full_body_outfit: bool = Field(default=False)


class Accessories(BaseModel):
    hat: str = Field(default="unknown", description="description or none")
    glasses: str = Field(default="unknown")
    collar: str = Field(default="unknown")
    ribbon: str = Field(default="unknown")
    other_visible_accessory: str = Field(default="unknown")


class Pose(BaseModel):
    posture: str = Field(default="unknown", description="standing / sitting / lying / jumping")
    facing_direction: str = Field(default="unknown")
    tail_position: str = Field(default="unknown")


class Environment(BaseModel):
    setting: str = Field(default="unknown", description="indoor / outdoor")
    dominant_background_colors: str = Field(default="unknown")


class ImageAnalysisResponse(BaseModel):
    """Structured visual attributes from image. JSON only; no commentary."""
    animal: AnimalInfo = Field(default_factory=AnimalInfo)
    clothing: ClothingDetection = Field(default_factory=ClothingDetection)
    accessories: Accessories = Field(default_factory=Accessories)
    pose: Pose = Field(default_factory=Pose)
    environment: Environment = Field(default_factory=Environment)


# ---------- Viewpoint / camera angle analysis (JSON only) ----------


class ViewpointAnalysisResponse(BaseModel):
    """Camera angle and subject orientation. JSON only."""
    view_angle: str = Field(
        default="three-quarter",
        description="front / three-quarter / side-profile-left / side-profile-right",
    )
    head_visible_eyes: int = Field(default=2, ge=1, le=2, description="1 or 2")
    body_orientation_degrees: int = Field(default=45, ge=0, le=180)
    tail_visible: bool = Field(default=False)


# ---------- Universal Animal Image Generation Engine (Phase 1 analysis) ----------


# High-precision clothing: only true if fabric texture visible; harness/collar alone does NOT count
CLOTHING_TYPES = ("sweater", "shirt", "hoodie", "dress", "costume", "none")
CLOTHING_COVERAGE = ("partial", "torso", "full-body", "none")


class UniversalAnalysisResponse(BaseModel):
    """
    Pose, camera, gravity, clothing, structure visibility.
    Do NOT auto-correct pose; detect only visible attributes.
    Clothing: fabric-based wearable covering torso/neck/legs/full body. Harness, leash, collar alone do NOT count.
    Only is_wearing_clothes=true if fabric texture visible; continuous fur without fabric boundary → false.
    """
    species: str = Field(default="unknown", description="dog | cat | other")
    view_angle: str = Field(
        default="three-quarter",
        description="front | three-quarter | side-left | side-right | rear",
    )
    body_pose: str = Field(
        default="unknown",
        description="standing | sitting | lying | crouching | jumping | unknown",
    )
    gravity_axis: str = Field(
        default="normal",
        description="normal | rotated-left | rotated-right | upside-down",
    )
    head_direction_degrees: int = Field(default=0, ge=0, le=360)
    spine_alignment: str = Field(
        default="vertical",
        description="vertical | horizontal | diagonal",
    )
    visible_eyes: int = Field(default=2, ge=0, le=2)
    leg_visibility_count: int = Field(default=4, ge=0, le=4)
    is_full_body_visible: bool = Field(default=True)
    is_wearing_clothes: bool = Field(
        default=False,
        description="True only if fabric texture visible; fur continuous without fabric boundary → false",
    )
    clothing_type: str = Field(
        default="none",
        description="sweater | shirt | hoodie | dress | costume | none",
    )
    clothing_coverage: str = Field(
        default="none",
        description="partial | torso | full-body | none",
    )
    fabric_texture_visible: bool = Field(
        default=False,
        description="Fabric boundary/texture detected; if false do not assume clothing",
    )
    clothing_color: str = Field(default="")
    clothing_pattern: str = Field(default="")
    clothing_confidence: float = Field(
        default=0.0,
        ge=0.0,
        le=1.0,
        description="If uncertain, lower confidence; do not assume clothing",
    )
