"""
Image generation API routes.
"""

from __future__ import annotations

import asyncio
import base64
import logging
import sys
import uuid
from typing import Annotated

import json as _json

import httpx
from fastapi import APIRouter, BackgroundTasks, Body, File, Form, Header, HTTPException, Request, UploadFile
from fastapi.responses import FileResponse, StreamingResponse

from app.core.config import get_settings
from app.models.style_presets import STYLE_PRESETS
from app.utils.prompt_builder import get_allowed_style_keys, get_allowed_species_keys, get_prompt_config
from app.schemas.image_schema import (
    ACBiologicalAnalysis,
    ACReconstructRequest,
    GenerateResponse,
    GenerateVideoResponse,
    ImageJobResponse,
    ImageJobStatusResponse,
    VideoJobResponse,
    VideoJobStatusResponse,
    ImageAnalysisResponse,
    AnimalInfo,
    ClothingDetection,
    Accessories,
    Pose,
    Environment,
    ViewpointAnalysisResponse,
    UniversalAnalysisResponse,
    CLOTHING_TYPES,
    CLOTHING_COVERAGE,
)
from app.services.image_service import (
    run_ac_villager_pipeline,
    run_ac_villager_reconstruct,
    run_image_to_image,
    run_universal_animal_generate,
)
from app.services.video_service import (
    run_image_to_video,
    _clamp_num_frames_to_8n_plus_1,
    DEFAULT_WIDTH,
    DEFAULT_HEIGHT,
    DEFAULT_NUM_FRAMES,
    DANCE_CONDITION_STRENGTH,
    DANCE_SHORT_WIDTH,
    DANCE_SHORT_HEIGHT,
    DANCE_SHORT_NUM_FRAMES,
    DANCE_SHORT_NUM_STEPS,
    DANCE_SHORT_GUIDANCE_SCALE,
    DANCE_SHORT_FRAME_RATE,
    NEGATIVE_PET_DANCE,
    QUALITY_WIDTH,
    QUALITY_HEIGHT,
    QUALITY_NUM_FRAMES,
    QUALITY_NUM_STEPS,
)
from app.services.training_store import (
    add_item as training_add_item,
    delete_item as training_delete_item,
    get_image_path as training_get_image_path,
    list_categories as training_list_categories,
    list_items as training_list_items,
    update_caption as training_update_caption,
    update_item as training_update_item,
)
from app.utils.file_handler import get_generated_url, save_upload_async
from app.schemas.dance_schema import DanceGenerateResponse, DanceJobResponse, DanceJobStatusResponse
from app.schemas.comfyui_schema import ComfyUIRunRequest, ComfyUIRunResponse
from app.services.comfyui_service import (
    run_workflow_and_get_image,
    run_workflow_save_to_generated,
    health_check as comfyui_health_check,
    _load_workflow_json as load_comfyui_workflow_json,
)
from app.services.dance_service import (
    get_motion_video_path,
    run_dance_generate,
    run_dance_generate_custom,
)

logger = logging.getLogger(__name__)

# 동영상 생성 비동기 잡 (1분 타임아웃 회피: POST는 즉시 반환, 클라이언트는 폴링)
_video_jobs: dict[str, dict] = {}
# 댄스 생성 비동기 잡
_dance_jobs: dict[str, dict] = {}

# 이미지 생성 비동기 잡 (62초+ 걸릴 때 타임아웃 회피: POST 즉시 반환, 클라이언트 폴링)
_image_jobs: dict[str, dict] = {}

# Dance Style: motion_id -> label and reference path (for UI)
DANCE_MOTIONS: list[dict[str, str]] = [
    {"id": "rat_dance", "label": "RAT Dance Challenge", "videoReference": "/motions/rat_dance.mp4"},
]

router = APIRouter(prefix="/api", tags=["api"])

# In-memory rate limit: ip -> (count, window_start). Use Redis in multi-worker production.
_rate_store: dict[str, tuple[int, float]] = {}
_CLEANUP_INTERVAL = 100  # clean old entries every N requests
_request_count = 0


def _check_rate_limit(request: Request) -> None:
    settings = get_settings()
    forwarded = request.headers.get("x-forwarded-for")
    ip = forwarded.split(",")[0].strip() if forwarded else (request.client.host or "unknown")
    import time
    now = time.time()
    global _rate_store, _request_count
    _request_count += 1
    if ip not in _rate_store:
        _rate_store[ip] = (1, now)
        return
    count, start = _rate_store[ip]
    if now - start > settings.rate_limit_window_seconds:
        _rate_store[ip] = (1, now)
        return
    if count >= settings.rate_limit_requests:
        raise HTTPException(status_code=429, detail="Too many requests")
    _rate_store[ip] = (count + 1, start)
    if _request_count % _CLEANUP_INTERVAL == 0:
        _rate_store = {k: v for k, v in _rate_store.items() if now - v[1] < settings.rate_limit_window_seconds}


def _parse_optional_float(s: str | None) -> float | None:
    if s is None or (isinstance(s, str) and not s.strip()):
        return None
    try:
        return float(s)
    except (ValueError, TypeError):
        return None


def _parse_optional_int(s: str | None) -> int | None:
    if s is None or (isinstance(s, str) and not s.strip()):
        return None
    try:
        return int(float(s))
    except (ValueError, TypeError):
        return None


@router.post("/generate", response_model=GenerateResponse)
async def generate(
    request: Request,
    file: Annotated[UploadFile | None, File(description="Image file (field: file or image)")] = None,
    image: Annotated[UploadFile | None, File(description="Image file (alias for file)")] = None,
    style: Annotated[str, Form(description="Style preset key")] = "pokemon",
    species: Annotated[str | None, Form(description="Pet species for silhouette/ear/eye rules: dog, cat, rabbit, hamster, ferret, bird, turtle, reptile, pet")] = None,
    ac_background: Annotated[str | None, Form(description="Animal Crossing style: background override. Omit for random.")] = None,
    ac_preserve_original: Annotated[str | None, Form(description="Animal Crossing: 'true' to preserve reference image composition, background, clothing, pose")] = None,
    ac_eye_color: Annotated[str | None, Form(description="Animal Crossing preserve mode: eye color e.g. amber, sapphire blue, black")] = None,
    ac_pose: Annotated[str | None, Form(description="Animal Crossing preserve mode: pose e.g. waving with one paw, holding fishing rod")] = None,
    ac_sign_text: Annotated[str | None, Form(description="Animal Crossing preserve mode: custom town sign text e.g. PAW-RADISE, MEOWTON")] = None,
    custom_prompt: Annotated[str | None, Form(description="Optional custom prompt")] = None,
    raw_prompt: Annotated[str | None, Form(description="If 'true'/'1'/'yes', use custom_prompt as-is")] = None,
    side_profile_lock: Annotated[str | None, Form(description="If 'true'/'1'/'yes', enforce 90° side profile (strength 0.65-0.75, guidance ≥8)")] = None,
    use_pose_lock: Annotated[str | None, Form(description="If 'true' and analysis provided, use Universal pose-lock engine")] = None,
    analysis: Annotated[str | None, Form(description="JSON from /api/image/analyze-universal for pose-lock generation")] = None,
    validate_and_retry: Annotated[str | None, Form(description="If 'true' with use_pose_lock, re-analyze output and retry on drift")] = None,
    steps: Annotated[str | None, Form(description="Inference steps (default 70)")] = None,
    strength: Annotated[str | None, Form()] = None,
    seed: Annotated[str | None, Form()] = None,
) -> GenerateResponse:
    """
    Upload image, run Z-Image-Turbo with selected style (and optional custom prompt).
    Returns URLs to original and generated images plus processing time.
    Accepts multipart field "file" or "image" (frontend may send "image").
    """
    _check_rate_limit(request)
    upload = file if (file and file.filename) else image
    if not upload or not upload.filename:
        raise HTTPException(status_code=422, detail="Missing required file. Send as multipart field 'file' or 'image'.")
    strength_f: float | None = _parse_optional_float(strength)
    seed_i: int | None = _parse_optional_int(seed)
    steps_i = _parse_optional_int(steps) if steps is not None else 9
    if steps_i is None or steps_i < 1 or steps_i > 70:
        steps_i = 9
    settings = get_settings()
    style_lower = style.strip().lower()
    allowed_list = get_allowed_style_keys()
    allowed = set(allowed_list) | {k.replace(" ", "_") for k in allowed_list}
    if style_lower not in allowed:
        raise HTTPException(
            status_code=400,
            detail=f"Invalid style. Allowed: {', '.join(sorted(allowed))}",
        )
    if not upload.content_type or not upload.content_type.startswith("image/"):
        raise HTTPException(status_code=400, detail="File must be an image (e.g. image/png, image/jpeg)")
    content = await upload.read()
    if len(content) > settings.upload_max_bytes:
        raise HTTPException(
            status_code=400,
            detail=f"File too large. Max {settings.upload_max_size_mb}MB",
        )
    if len(content) == 0:
        raise HTTPException(status_code=400, detail="Empty file")
    # Save original to static/generated
    try:
        original_filename = await save_upload_async(content, suffix=".png")
    except Exception as e:
        logger.exception("Failed to save upload: %s", e)
        raise HTTPException(status_code=500, detail="Failed to save uploaded image")
    original_url = get_generated_url(original_filename)
    raw_prompt_bool = (raw_prompt or "").strip().lower() in ("true", "1", "yes")
    use_pose_lock_bool = (use_pose_lock or "").strip().lower() in ("true", "1", "yes")
    validate_and_retry_bool = (validate_and_retry or "").strip().lower() in ("true", "1", "yes")
    analysis_dict = None
    if use_pose_lock_bool and analysis and analysis.strip():
        try:
            analysis_dict = _json.loads(analysis.strip())
        except _json.JSONDecodeError as e:
            raise HTTPException(status_code=400, detail=f"Invalid analysis JSON: {e}") from e

    print("[이미지 생성] 로컬에서 생성 중...", file=sys.stderr, flush=True)
    try:
        if use_pose_lock_bool and analysis_dict is not None:
            out_bytes, processing_time = await run_universal_animal_generate(
                image_bytes=content,
                analysis=analysis_dict,
                style_key=style_lower,
                species_key=species.strip().lower() if species and species.strip() else None,
                seed=seed_i,
                validate_and_retry=validate_and_retry_bool,
                max_retries=1,
            )
        else:
            out_bytes, processing_time = await run_image_to_image(
                image_bytes=content,
                style_key=style_lower,
                species_key=species.strip().lower() if species and species.strip() else None,
                custom_prompt=custom_prompt,
                raw_prompt=raw_prompt_bool,
                num_steps=steps_i,
                strength=strength_f,
                seed=seed_i,
                ac_background=ac_background.strip() if ac_background and ac_background.strip() else None,
                ac_preserve_original=(ac_preserve_original or "").strip().lower() in ("true", "1", "yes"),
                ac_eye_color=ac_eye_color.strip() if ac_eye_color and ac_eye_color.strip() else None,
                ac_pose=ac_pose.strip() if ac_pose and ac_pose.strip() else None,
                ac_sign_text=ac_sign_text.strip() if ac_sign_text and ac_sign_text.strip() else None,
                side_profile_lock=(side_profile_lock or "").strip().lower() in ("true", "1", "yes"),
            )
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except RuntimeError as e:
        logger.exception("Model inference failed: %s", e)
        raise HTTPException(status_code=503, detail=str(e))
    print("[이미지 생성] 로컬 생성 완료 (%.1f초)" % processing_time, file=sys.stderr, flush=True)
    try:
        generated_filename = await save_upload_async(out_bytes, suffix=".png")
    except Exception as e:
        logger.exception("Failed to save generated image: %s", e)
        raise HTTPException(status_code=500, detail="Failed to save generated image")
    generated_url = get_generated_url(generated_filename)
    # 멀티 Pod/K8s: GET /static/generated/xxx 가 다른 Pod로 가면 404. 응답에 base64 포함해 두 번째 요청 없이 표시 가능하게 함.
    generated_b64 = base64.b64encode(out_bytes).decode("ascii") if out_bytes else None
    return GenerateResponse(
        original_url=original_url,
        generated_url=generated_url,
        processing_time=round(processing_time, 2),
        generated_image_base64=generated_b64,
    )


async def _run_image_job(job_id: str, payload: dict) -> None:
    """백그라운드에서 이미지 생성 후 잡 상태 갱신. 60초+ 타임아웃 회피용."""
    try:
        content = payload["content"]
        style_lower = payload["style_lower"]
        original_url = payload["original_url"]
        use_pose_lock_bool = payload.get("use_pose_lock_bool", False)
        analysis_dict = payload.get("analysis_dict")
        validate_and_retry_bool = payload.get("validate_and_retry_bool", False)
        if use_pose_lock_bool and analysis_dict is not None:
            out_bytes, processing_time = await run_universal_animal_generate(
                image_bytes=content,
                analysis=analysis_dict,
                style_key=style_lower,
                species_key=payload.get("species_key"),
                seed=payload.get("seed_i"),
                validate_and_retry=validate_and_retry_bool,
                max_retries=1,
            )
        else:
            out_bytes, processing_time = await run_image_to_image(
                image_bytes=content,
                style_key=style_lower,
                species_key=payload.get("species_key"),
                custom_prompt=payload.get("custom_prompt"),
                raw_prompt=payload.get("raw_prompt_bool", False),
                num_steps=payload.get("steps_i", 70),
                strength=payload.get("strength_f"),
                seed=payload.get("seed_i"),
                ac_background=payload.get("ac_background"),
                ac_preserve_original=payload.get("ac_preserve_original", False),
                ac_eye_color=payload.get("ac_eye_color"),
                ac_pose=payload.get("ac_pose"),
                ac_sign_text=payload.get("ac_sign_text"),
                side_profile_lock=payload.get("side_profile_lock", False),
            )
        generated_filename = await save_upload_async(out_bytes, suffix=".png")
        generated_url = get_generated_url(generated_filename)
        generated_b64 = base64.b64encode(out_bytes).decode("ascii") if out_bytes else None
        _image_jobs[job_id] = {
            "status": "completed",
            "original_url": original_url,
            "generated_url": generated_url,
            "processing_time": round(processing_time, 2),
            "generated_image_base64": generated_b64,
            "error": None,
        }
    except Exception as e:
        logger.exception("Image job %s failed: %s", job_id, e)
        _image_jobs[job_id] = {
            "status": "failed",
            "original_url": payload.get("original_url"),
            "generated_url": None,
            "processing_time": None,
            "generated_image_base64": None,
            "error": str(e),
        }


@router.post("/image/generate", response_model=ImageJobResponse)
async def generate_image_async(
    request: Request,
    file: Annotated[UploadFile | None, File(description="Image file")] = None,
    image: Annotated[UploadFile | None, File(description="Image file (alias)")] = None,
    style: Annotated[str, Form(description="Style preset key")] = "sailor_moon",
    species: Annotated[str | None, Form(description="Pet species")] = None,
    ac_background: Annotated[str | None, Form()] = None,
    ac_preserve_original: Annotated[str | None, Form()] = None,
    ac_eye_color: Annotated[str | None, Form()] = None,
    ac_pose: Annotated[str | None, Form()] = None,
    ac_sign_text: Annotated[str | None, Form()] = None,
    custom_prompt: Annotated[str | None, Form()] = None,
    raw_prompt: Annotated[str | None, Form()] = None,
    side_profile_lock: Annotated[str | None, Form()] = None,
    use_pose_lock: Annotated[str | None, Form()] = None,
    analysis: Annotated[str | None, Form()] = None,
    validate_and_retry: Annotated[str | None, Form()] = None,
    steps: Annotated[str | None, Form()] = None,
    strength: Annotated[str | None, Form()] = None,
    seed: Annotated[str | None, Form()] = None,
) -> ImageJobResponse:
    """
    이미지 생성 비동기. 즉시 job_id 반환. GET /api/image/status/{job_id} 로 폴링 (62초+ 걸릴 때 타임아웃 회피).
    """
    _check_rate_limit(request)
    upload = file if (file and file.filename) else image
    if not upload or not upload.filename:
        raise HTTPException(status_code=422, detail="Missing image file. Send as 'file' or 'image'.")
    content = await upload.read()
    settings = get_settings()
    if len(content) > settings.upload_max_bytes:
        raise HTTPException(status_code=400, detail=f"File too large. Max {settings.upload_max_size_mb}MB")
    if len(content) == 0:
        raise HTTPException(status_code=400, detail="Empty file")
    if not upload.content_type or not upload.content_type.startswith("image/"):
        raise HTTPException(status_code=400, detail="File must be an image")
    style_lower = style.strip().lower()
    allowed_list = get_allowed_style_keys()
    allowed = set(allowed_list) | {k.replace(" ", "_") for k in allowed_list}
    if style_lower not in allowed:
        raise HTTPException(status_code=400, detail=f"Invalid style. Allowed: {', '.join(sorted(allowed))}")
    try:
        original_filename = await save_upload_async(content, suffix=".png")
    except Exception as e:
        logger.exception("Failed to save upload: %s", e)
        raise HTTPException(status_code=500, detail="Failed to save uploaded image")
    original_url = get_generated_url(original_filename)
    raw_prompt_bool = (raw_prompt or "").strip().lower() in ("true", "1", "yes")
    use_pose_lock_bool = (use_pose_lock or "").strip().lower() in ("true", "1", "yes")
    validate_and_retry_bool = (validate_and_retry or "").strip().lower() in ("true", "1", "yes")
    analysis_dict = None
    if use_pose_lock_bool and analysis and analysis.strip():
        try:
            analysis_dict = _json.loads(analysis.strip())
        except _json.JSONDecodeError as e:
            raise HTTPException(status_code=400, detail=f"Invalid analysis JSON: {e}") from e
    steps_i = _parse_optional_int(steps) if steps is not None else 9
    if steps_i is None or steps_i < 1 or steps_i > 70:
        steps_i = 9
    payload = {
        "content": content,
        "style_lower": style_lower,
        "original_url": original_url,
        "species_key": species.strip().lower() if species and species.strip() else None,
        "custom_prompt": custom_prompt,
        "raw_prompt_bool": raw_prompt_bool,
        "steps_i": steps_i,
        "strength_f": _parse_optional_float(strength),
        "seed_i": _parse_optional_int(seed),
        "ac_background": ac_background.strip() if ac_background and ac_background.strip() else None,
        "ac_preserve_original": (ac_preserve_original or "").strip().lower() in ("true", "1", "yes"),
        "ac_eye_color": ac_eye_color.strip() if ac_eye_color and ac_eye_color.strip() else None,
        "ac_pose": ac_pose.strip() if ac_pose and ac_pose.strip() else None,
        "ac_sign_text": ac_sign_text.strip() if ac_sign_text and ac_sign_text.strip() else None,
        "side_profile_lock": (side_profile_lock or "").strip().lower() in ("true", "1", "yes"),
        "use_pose_lock_bool": use_pose_lock_bool,
        "analysis_dict": analysis_dict,
        "validate_and_retry_bool": validate_and_retry_bool,
    }
    job_id = str(uuid.uuid4())
    _image_jobs[job_id] = {
        "status": "processing",
        "original_url": None,
        "generated_url": None,
        "processing_time": None,
        "generated_image_base64": None,
        "error": None,
    }
    asyncio.create_task(_run_image_job(job_id, payload))
    return ImageJobResponse(job_id=job_id)


@router.get("/image/status/{job_id}", response_model=ImageJobStatusResponse)
async def get_image_job_status(job_id: str) -> ImageJobStatusResponse:
    """이미지 생성 잡 상태. processing → 2~3초마다 폴링, completed 시 URL·base64 반환."""
    if job_id not in _image_jobs:
        raise HTTPException(status_code=404, detail="Job not found or expired")
    job = _image_jobs[job_id]
    return ImageJobStatusResponse(
        status=job["status"],
        original_url=job.get("original_url"),
        generated_url=job.get("generated_url"),
        processing_time=job.get("processing_time"),
        generated_image_base64=job.get("generated_image_base64"),
        error=job.get("error"),
    )


@router.post("/generate/ac-villager", response_model=GenerateResponse)
async def generate_ac_villager(
    request: Request,
    file: Annotated[UploadFile | None, File(description="Pet image (required)")] = None,
    image: Annotated[UploadFile | None, File(description="Pet image (alias)")] = None,
    species: Annotated[str | None, Form(description="Stage 1: cat, dog, rabbit, hamster, bird, other. Omit for 'other'.")] = None,
    main_color: Annotated[str | None, Form(description="Stage 3: main fur color (e.g. cream, orange)")] = None,
    secondary_color: Annotated[str | None, Form(description="Stage 3: secondary fur color")] = None,
    eye_color: Annotated[str | None, Form(description="Stage 3: eye color (e.g. amber, green)")] = None,
    markings: Annotated[str | None, Form(description="Stage 3: major markings (e.g. white chest, stripes)")] = None,
    seed: Annotated[str | None, Form()] = None,
) -> GenerateResponse:
    """
    4-stage Animal Crossing villager pipeline:
    Stage 1: species from form or 'other'.
    Stage 2: base villager (strict AC proportions) via high-strength img2img from gray.
    Stage 3: traits from form or defaults (plug in vision extraction later).
    Stage 4: color transfer only (strength 0.35-0.42) to apply fur/eye/markings.
    """
    _check_rate_limit(request)
    upload = file if (file and file.filename) else image
    if not upload or not upload.filename:
        raise HTTPException(status_code=422, detail="Missing required file. Send as 'file' or 'image'.")
    settings = get_settings()
    if not upload.content_type or not upload.content_type.startswith("image/"):
        raise HTTPException(status_code=400, detail="File must be an image.")
    content = await upload.read()
    if len(content) > settings.upload_max_bytes:
        raise HTTPException(status_code=400, detail=f"File too large. Max {settings.upload_max_size_mb}MB")
    if len(content) == 0:
        raise HTTPException(status_code=400, detail="Empty file")
    try:
        original_filename = await save_upload_async(content, suffix=".png")
    except Exception as e:
        logger.exception("Failed to save upload: %s", e)
        raise HTTPException(status_code=500, detail="Failed to save uploaded image")
    original_url = get_generated_url(original_filename)
    seed_i = int(seed) if seed is not None and str(seed).strip().lstrip("-").isdigit() else None
    try:
        out_bytes, processing_time = await run_ac_villager_pipeline(
            image_bytes=content,
            species_override=species.strip() if species and species.strip() else None,
            main_color=main_color,
            secondary_color=secondary_color,
            eye_color=eye_color,
            markings=markings,
            seed=seed_i,
        )
    except RuntimeError as e:
        logger.exception("AC villager pipeline failed: %s", e)
        raise HTTPException(status_code=503, detail=str(e))
    try:
        generated_filename = await save_upload_async(out_bytes, suffix=".png")
    except Exception as e:
        logger.exception("Failed to save generated image: %s", e)
        raise HTTPException(status_code=500, detail="Failed to save generated image")
    generated_url = get_generated_url(generated_filename)
    generated_b64 = base64.b64encode(out_bytes).decode("ascii") if out_bytes else None
    return GenerateResponse(
        original_url=original_url,
        generated_url=generated_url,
        processing_time=round(processing_time, 2),
        generated_image_base64=generated_b64,
    )


# ---------- AC Villager Reconstruction (Stage 1: analyze, Stage 2: T2I-only) ----------


@router.post("/ac/analyze", response_model=ACBiologicalAnalysis)
async def ac_analyze(
    request: Request,
    file: Annotated[UploadFile | None, File(description="Pet image for biological analysis")] = None,
    image: Annotated[UploadFile | None, File(description="Alias for file")] = None,
    species: Annotated[str | None, Form(description="Override: cat, dog, rabbit, hamster, bird")] = None,
    main_fur_color: Annotated[str | None, Form()] = None,
    secondary_fur_color: Annotated[str | None, Form()] = None,
    eye_color: Annotated[str | None, Form()] = None,
    markings: Annotated[str | None, Form()] = None,
    ear_type: Annotated[str | None, Form()] = None,
    tail_type: Annotated[str | None, Form()] = None,
) -> ACBiologicalAnalysis:
    """
    Stage 1 — Biological analysis. Extract species, fur colors, eye color, markings, ear/tail type from image.
    Returns structured data only; no rendering. If no vision model is wired, form overrides or defaults are used.
    """
    _check_rate_limit(request)
    # Stub: no vision model yet; use form values or defaults. Plug in vision/LLM here later.
    species_val = (species or "cat").strip().lower()
    if species_val not in ("cat", "dog", "rabbit", "hamster", "bird", "other"):
        species_val = "cat"
    ear_defaults = {"cat": "pointed triangular ears", "dog": "rounded ears", "rabbit": "long upright ears", "hamster": "small round ears", "bird": "no external ears", "other": "simplified ears"}
    tail_defaults = {"cat": "simplified tapered tail", "dog": "short stylized tail", "rabbit": "short cotton tail", "hamster": "tiny round tail", "bird": "short tail feathers", "other": "simplified tail"}
    return ACBiologicalAnalysis(
        species=species_val,
        main_fur_color=(main_fur_color or "cream").strip(),
        secondary_fur_color=(secondary_fur_color or "none").strip(),
        eye_color=(eye_color or "amber").strip(),
        markings=(markings or "none").strip(),
        ear_type=(ear_type or ear_defaults.get(species_val, "simplified ears")).strip(),
        tail_type=(tail_type or tail_defaults.get(species_val, "simplified tail")).strip(),
    )


@router.post("/generate/ac-villager-reconstruct", response_model=GenerateResponse)
async def generate_ac_villager_reconstruct(
    request: Request,
    body: ACReconstructRequest,
) -> GenerateResponse:
    """
    Stage 2 — Villager reconstruction. TEXT-TO-IMAGE ONLY (no img2img from user image).
    Uses biological data from Stage 1 (or manual). Strict Nintendo AC proportions, full environment replacement.
    """
    _check_rate_limit(request)
    species_key = (body.species or "other").strip().lower()
    if species_key not in ("cat", "dog", "rabbit", "hamster", "bird", "other"):
        species_key = "other"
    try:
        out_bytes, processing_time = await run_ac_villager_reconstruct(
            species=species_key,
            main_fur_color=body.main_fur_color,
            secondary_fur_color=body.secondary_fur_color,
            eye_color=body.eye_color,
            markings=body.markings,
            ear_type=body.ear_type,
            tail_type=body.tail_type,
            seed=body.seed,
        )
    except RuntimeError as e:
        logger.exception("AC villager reconstruct failed: %s", e)
        raise HTTPException(status_code=503, detail=str(e))
    try:
        generated_filename = await save_upload_async(out_bytes, suffix=".png")
    except Exception as e:
        logger.exception("Failed to save generated image: %s", e)
        raise HTTPException(status_code=500, detail="Failed to save generated image")
    generated_url = get_generated_url(generated_filename)
    generated_b64 = base64.b64encode(out_bytes).decode("ascii") if out_bytes else None
    return GenerateResponse(
        original_url=generated_url,
        generated_url=generated_url,
        processing_time=round(processing_time, 2),
        generated_image_base64=generated_b64,
    )


# ---------- Image Analysis (structured visual attributes, JSON only) ----------


@router.post("/image/viewpoint", response_model=ViewpointAnalysisResponse)
async def analyze_viewpoint(
    request: Request,
    file: Annotated[UploadFile | None, File(description="Image to analyze")] = None,
    image: Annotated[UploadFile | None, File(description="Alias for file")] = None,
    view_angle: Annotated[str | None, Form(description="Override: front / three-quarter / side-profile-left / side-profile-right")] = None,
    head_visible_eyes: Annotated[str | None, Form(description="Override: 1 or 2")] = None,
    body_orientation_degrees: Annotated[str | None, Form(description="Override: 0-180")] = None,
    tail_visible: Annotated[str | None, Form(description="Override: true/false")] = None,
) -> ViewpointAnalysisResponse:
    """
    Camera angle and subject orientation. JSON only.
    Stub: when no vision model is wired, uses form overrides or defaults.
    If only one eye is visible, classify as side-profile.
    """
    _check_rate_limit(request)
    angle = (view_angle or "three-quarter").strip().lower()
    if angle not in ("front", "three-quarter", "side-profile-left", "side-profile-right"):
        angle = "three-quarter"
    eyes = 1 if angle.startswith("side-profile") else 2
    if head_visible_eyes is not None and str(head_visible_eyes).strip() in ("1", "2"):
        eyes = int(head_visible_eyes)
    deg = 90 if angle.startswith("side-profile") else (45 if angle == "three-quarter" else 0)
    if body_orientation_degrees is not None:
        try:
            d = int(float(body_orientation_degrees))
            deg = max(0, min(180, d))
        except (ValueError, TypeError):
            pass
    tail = (tail_visible or "").strip().lower() in ("true", "1", "yes")
    return ViewpointAnalysisResponse(
        view_angle=angle,
        head_visible_eyes=eyes,
        body_orientation_degrees=deg,
        tail_visible=tail,
    )


@router.post("/image/analyze-universal", response_model=UniversalAnalysisResponse)
async def analyze_universal(
    request: Request,
    file: Annotated[UploadFile | None, File(description="Image to analyze")] = None,
    image: Annotated[UploadFile | None, File(description="Alias for file")] = None,
    species: Annotated[str | None, Form()] = None,
    view_angle: Annotated[str | None, Form(description="front | three-quarter | side-left | side-right | rear")] = None,
    body_pose: Annotated[str | None, Form(description="standing | sitting | lying | crouching | jumping | unknown")] = None,
    gravity_axis: Annotated[str | None, Form(description="normal | rotated-left | rotated-right | upside-down")] = None,
    head_direction_degrees: Annotated[str | None, Form()] = None,
    spine_alignment: Annotated[str | None, Form(description="vertical | horizontal | diagonal")] = None,
    visible_eyes: Annotated[str | None, Form(description="0-2")] = None,
    leg_visibility_count: Annotated[str | None, Form(description="0-4")] = None,
    is_full_body_visible: Annotated[str | None, Form(description="true/false")] = None,
    is_wearing_clothes: Annotated[str | None, Form(description="true/false")] = None,
    clothing_type: Annotated[str | None, Form(description="sweater | shirt | hoodie | dress | costume | none")] = None,
    clothing_coverage: Annotated[str | None, Form(description="partial | torso | full-body | none")] = None,
    fabric_texture_visible: Annotated[str | None, Form(description="true/false")] = None,
    clothing_color: Annotated[str | None, Form()] = None,
    clothing_pattern: Annotated[str | None, Form()] = None,
    clothing_confidence: Annotated[str | None, Form(description="0.0-1.0")] = None,
) -> UniversalAnalysisResponse:
    """
    Universal analysis: pose, camera, gravity, high-precision clothing.
    Only is_wearing_clothes=true if fabric texture visible. Harness/collar alone do NOT count.
    """
    _check_rate_limit(request)
    def _int(s: str | None, lo: int, hi: int, default: int) -> int:
        if s is None or not str(s).strip():
            return default
        try:
            v = int(float(s))
            return max(lo, min(hi, v))
        except (ValueError, TypeError):
            return default
    def _float(s: str | None, lo: float, hi: float, default: float) -> float:
        if s is None or not str(s).strip():
            return default
        try:
            v = float(s)
            return max(lo, min(hi, v))
        except (ValueError, TypeError):
            return default
    va = (view_angle or "three-quarter").strip().lower()
    if va not in ("front", "three-quarter", "side-left", "side-right", "rear"):
        va = "three-quarter"
    bp = (body_pose or "unknown").strip().lower()
    if bp not in ("standing", "sitting", "lying", "crouching", "jumping", "unknown"):
        bp = "unknown"
    grav = (gravity_axis or "normal").strip().lower()
    if grav not in ("normal", "rotated-left", "rotated-right", "upside-down"):
        grav = "normal"
    spine = (spine_alignment or "vertical").strip().lower()
    if spine not in ("vertical", "horizontal", "diagonal"):
        spine = "vertical"
    # High-precision clothing: only true if fabric texture visible; if uncertain do not assume
    fabric_visible = (fabric_texture_visible or "").strip().lower() in ("true", "1", "yes")
    form_says_clothes = (is_wearing_clothes or "").strip().lower() in ("true", "1", "yes")
    is_clothes = fabric_visible if (fabric_texture_visible is not None and str(fabric_texture_visible).strip()) else form_says_clothes
    ct = (clothing_type or "none").strip().lower()
    if ct not in CLOTHING_TYPES:
        ct = "none" if not is_clothes else "sweater"
    cov = (clothing_coverage or "none").strip().lower()
    if cov not in CLOTHING_COVERAGE:
        cov = "none" if not is_clothes else "torso"
    conf = _float(clothing_confidence, 0.0, 1.0, 0.8 if is_clothes else 0.0)
    return UniversalAnalysisResponse(
        species=(species or "unknown").strip(),
        view_angle=va,
        body_pose=bp,
        gravity_axis=grav,
        head_direction_degrees=_int(head_direction_degrees, 0, 360, 0),
        spine_alignment=spine,
        visible_eyes=_int(visible_eyes, 0, 2, 2),
        leg_visibility_count=_int(leg_visibility_count, 0, 4, 4),
        is_full_body_visible=(is_full_body_visible or "").strip().lower() not in ("false", "0", "no"),
        is_wearing_clothes=is_clothes,
        clothing_type=ct,
        clothing_coverage=cov,
        fabric_texture_visible=fabric_visible,
        clothing_color=(clothing_color or "").strip(),
        clothing_pattern=(clothing_pattern or "").strip(),
        clothing_confidence=conf,
    )


@router.post("/image/analyze", response_model=ImageAnalysisResponse)
async def analyze_image(
    request: Request,
    file: Annotated[UploadFile | None, File(description="Image to analyze")] = None,
    image: Annotated[UploadFile | None, File(description="Alias for file")] = None,
    species: Annotated[str | None, Form(description="Override: cat / dog / other")] = None,
    fur_main_color: Annotated[str | None, Form()] = None,
    fur_secondary_color: Annotated[str | None, Form()] = None,
    major_markings: Annotated[str | None, Form()] = None,
    is_wearing_clothes: Annotated[str | None, Form(description="true/false")] = None,
    clothing_type: Annotated[str | None, Form()] = None,
    clothing_color: Annotated[str | None, Form()] = None,
    posture: Annotated[str | None, Form(description="standing/sitting/lying/jumping")] = None,
) -> ImageAnalysisResponse:
    """
    Extract structured visual attributes from image. Returns JSON only; no commentary.
    Stub: when no vision model is wired, uses form overrides or defaults to unknown.
    """
    _check_rate_limit(request)
    # Stub: no vision model; use form overrides or unknowns. Plug in vision/LLM here later.
    species_val = (species or "other").strip().lower()
    if species_val not in ("cat", "dog", "other"):
        species_val = "other"
    return ImageAnalysisResponse(
        animal=AnimalInfo(
            species=species_val,
            breed=None,
            fur_main_color=(fur_main_color or "unknown").strip(),
            fur_secondary_color=(fur_secondary_color or "unknown").strip(),
            major_markings=(major_markings or "unknown").strip(),
        ),
        clothing=ClothingDetection(
            is_wearing_clothes=(is_wearing_clothes or "").strip().lower() in ("true", "1", "yes"),
            clothing_type=(clothing_type or "unknown").strip(),
            clothing_color=(clothing_color or "unknown").strip(),
            clothing_pattern="unknown",
            sleeve_length="unknown",
            full_body_outfit=False,
        ),
        accessories=Accessories(hat="unknown", glasses="unknown", collar="unknown", ribbon="unknown", other_visible_accessory="unknown"),
        pose=Pose(
            posture=(posture or "unknown").strip().lower() if posture else "unknown",
            facing_direction="unknown",
            tail_position="unknown",
        ),
        environment=Environment(setting="unknown", dominant_background_colors="unknown"),
    )


@router.get("/styles")
async def list_styles() -> dict[str, str]:
    """Return available style presets (key -> description). prompt_builder와 동기화된 키만 노출."""
    allowed = get_allowed_style_keys()
    return {k: STYLE_PRESETS.get(k, k) for k in allowed}


@router.get("/prompt-config")
async def get_prompt_config_route() -> dict:
    """Return full prompt config: base_prompt, base_negative, species_modifiers, styles, generation_rules."""
    return get_prompt_config()


@router.get("/species")
async def list_species() -> dict[str, str]:
    """Return species modifiers (key -> silhouette/ear/eye/tail rules)."""
    from app.utils.prompt_builder import SPECIES_MODIFIERS
    return dict(SPECIES_MODIFIERS)


# ---------- LTX-2 Image-to-Video ----------
# 테스트 프리셋 + 트렌디 스타일 (6W·시네마틱 구조)
VIDEO_PROMPT_PRESETS: dict[str, str] = {
    "smile_turn": (
        "A fixed camera medium shot. The person starts facing slightly away, then slowly begins turning their head toward the camera. "
        "The turn continues smoothly; eyes blink naturally, a subtle smile appears, and hair or skin shifts with the motion. "
        "The head and neck keep rotating until facing the lens. Body and background stay still; camera stays locked. No pan, no tilt, no zoom."
    ),
    "wind_leaves": (
        "Fixed camera, locked frame. Leaves and branches begin moving in the wind: they sway, then rustle, then bend slightly. "
        "The motion continues rhythmically; foliage keeps drifting and swaying. Person or subject in frame stays still. Camera does not move."
    ),
    # 반려동물: dance 단어 제거 → shift weight + very small steps. motion strength 0.42, guidance 2.3.
    "dancing_pet": (
        "a cute dog standing on the ground, full body visible. "
        "the dog gently shifts its body weight left and right, making very small steps in place. "
        "its tail wagging slowly. "
        "slow natural dog movement, simple repeating motion. "
        "consistent dog appearance across frames, stable head and body. "
        "static camera, fixed framing."
    ),
    # 트렌디 스타일 4종
    "golden_hour": (
        "Fixed camera medium shot in warm golden hour sunlight. Soft, directional light creates long shadows and a glowing rim. "
        "The subject begins with a small breath, then hair or fabric moves gently in the breeze; the motion continues with subtle shifts. "
        "Film grain and warm color palette. Camera stays locked; no pan or zoom."
    ),
    "cozy_moment": (
        "Fixed camera. Cozy indoor scene with soft diffused lighting, warm tones, and a calm atmosphere. "
        "The subject starts relaxed, then slowly blinks, shifts posture, or makes a small gesture; the movement repeats gently. "
        "Shallow depth of field, soft bokeh in background. Camera does not move."
    ),
    "neon_night": (
        "Fixed camera. Night scene with neon reflections on wet pavement and soft fog in the air. "
        "The subject begins still, then breathes, turns their head slightly, or their eyes catch the neon glow; the motion continues. "
        "Cinematic color grading, cyan and magenta highlights. Camera stays locked."
    ),
    "dreamy_bokeh": (
        "Fixed camera. Dreamy soft-focus shot with shallow depth of field and creamy bokeh in the background. "
        "The subject starts neutral, then blinks softly, smiles slightly, or slowly turns toward camera; the motion continues gently. "
        "Ethereal atmosphere, fine film grain. Camera stays locked; no pan or zoom."
    ),
}


@router.get("/video/presets")
async def list_video_presets() -> dict[str, str]:
    """동영상 생성용 프롬프트 프리셋 (테스트 + 트렌디 스타일)."""
    return dict(VIDEO_PROMPT_PRESETS)


async def _run_video_job(job_id: str, run_kw: dict) -> None:
    """백그라운드에서 동영상 생성 후 잡 상태 갱신. 1분 타임아웃 회피용."""
    try:
        out_bytes, processing_time = await run_image_to_video(**run_kw)
        video_filename = await save_upload_async(out_bytes, suffix=".mp4")
        video_url = get_generated_url(video_filename)
        generated_b64 = base64.b64encode(out_bytes).decode("ascii") if out_bytes else None
        _video_jobs[job_id] = {
            "status": "completed",
            "video_url": video_url,
            "processing_time": round(processing_time, 2),
            "video_base64": generated_b64,
            "error": None,
        }
    except Exception as e:
        logger.exception("Video job %s failed: %s", job_id, e)
        _video_jobs[job_id] = {
            "status": "failed",
            "video_url": None,
            "processing_time": None,
            "video_base64": None,
            "error": str(e),
        }


@router.post("/video/generate", response_model=VideoJobResponse)
async def generate_video(
    request: Request,
    file: Annotated[UploadFile | None, File(description="Image file")] = None,
    image: Annotated[UploadFile | None, File(description="Image file (alias)")] = None,
    prompt: Annotated[str, Form(description="동영상 동작/장면 설명 (필수)")] = "",
    preset: Annotated[str | None, Form(description="프리셋 키: smile_turn, wind_leaves, dancing_pet, golden_hour, cozy_moment, neon_night, dreamy_bokeh")] = None,
    negative_prompt: Annotated[str | None, Form()] = None,
    num_frames: Annotated[str | None, Form()] = None,
    num_inference_steps: Annotated[str | None, Form()] = None,
    seed: Annotated[str | None, Form()] = None,
) -> VideoJobResponse:
    """
    LTX-2 이미지→동영상. 즉시 job_id 반환. GET /api/video/status/{job_id} 로 폴링하여 결과 조회 (1분 끊김 회피).
    """
    _check_rate_limit(request)
    upload = file if (file and file.filename) else image
    if not upload or not upload.filename:
        raise HTTPException(status_code=422, detail="Missing image file. Send as 'file' or 'image'.")
    if not (prompt or "").strip() and not preset:
        raise HTTPException(status_code=400, detail="prompt 또는 preset을 입력해주세요.")
    if preset and preset.strip() and preset.strip() in VIDEO_PROMPT_PRESETS:
        prompt = VIDEO_PROMPT_PRESETS[preset.strip()]
    else:
        prompt = (prompt or "").strip() or list(VIDEO_PROMPT_PRESETS.values())[0]
    content = await upload.read()
    if len(content) == 0:
        raise HTTPException(status_code=400, detail="Empty file")
    if not upload.content_type or not upload.content_type.startswith("image/"):
        raise HTTPException(status_code=400, detail="File must be an image (e.g. image/png, image/jpeg)")
    settings = get_settings()
    if len(content) > settings.upload_max_bytes:
        raise HTTPException(status_code=400, detail=f"File too large. Max {settings.upload_max_size_mb}MB")
    quality_mode = getattr(settings, "ltx2_quality_mode", False)
    use_dance_short = preset and preset.strip() == "dancing_pet"
    num_f = _parse_optional_int(num_frames)
    if num_f is None or num_f < 1:
        if use_dance_short:
            num_f = DANCE_SHORT_NUM_FRAMES
        else:
            num_f = QUALITY_NUM_FRAMES if quality_mode else DEFAULT_NUM_FRAMES
    num_f = _clamp_num_frames_to_8n_plus_1(
        min(241, max(25 if use_dance_short else 33, num_f))
    )
    steps = _parse_optional_int(num_inference_steps)
    if steps is None or steps < 1:
        steps = DANCE_SHORT_NUM_STEPS if use_dance_short else (QUALITY_NUM_STEPS if quality_mode else 25)
    steps = min(50, max(8, steps)) if use_dance_short else min(50, max(10, steps))
    width = DANCE_SHORT_WIDTH if use_dance_short else (QUALITY_WIDTH if quality_mode else DEFAULT_WIDTH)
    height = DANCE_SHORT_HEIGHT if use_dance_short else (QUALITY_HEIGHT if quality_mode else DEFAULT_HEIGHT)
    neg = (negative_prompt or "").strip()
    if use_dance_short and NEGATIVE_PET_DANCE:
        neg = f"{neg} {NEGATIVE_PET_DANCE}".strip() if neg else NEGATIVE_PET_DANCE
    else:
        neg = negative_prompt
    seed_i = _parse_optional_int(seed)
    run_kw: dict = {
        "image_bytes": content,
        "prompt": prompt,
        "negative_prompt": neg,
        "width": width,
        "height": height,
        "num_frames": num_f,
        "num_inference_steps": steps,
        "seed": seed_i,
    }
    if use_dance_short:
        run_kw["guidance_scale"] = DANCE_SHORT_GUIDANCE_SCALE
        run_kw["frame_rate"] = DANCE_SHORT_FRAME_RATE
        run_kw["condition_strength"] = DANCE_CONDITION_STRENGTH

    job_id = str(uuid.uuid4())
    _video_jobs[job_id] = {"status": "processing", "video_url": None, "processing_time": None, "video_base64": None, "error": None}
    asyncio.create_task(_run_video_job(job_id, run_kw))
    return VideoJobResponse(job_id=job_id)


@router.get("/video/status/{job_id}", response_model=VideoJobStatusResponse)
async def get_video_job_status(job_id: str) -> VideoJobStatusResponse:
    """동영상 생성 잡 상태. processing → 5초마다 폴링, completed 시 video_url 등 반환."""
    if job_id not in _video_jobs:
        raise HTTPException(status_code=404, detail="Job not found or expired")
    job = _video_jobs[job_id]
    return VideoJobStatusResponse(
        status=job["status"],
        video_url=job.get("video_url"),
        processing_time=job.get("processing_time"),
        video_base64=job.get("video_base64"),
        error=job.get("error"),
    )


@router.get("/video/comfyui/workflow")
async def get_comfyui_video_workflow():
    """
    ComfyUI LTX 이미지→비디오 워크플로 JSON 반환.
    프론트에서 ComfyUI API 직접 호출 시 사용 (업로드·prompt 주입·폴링·view는 프론트에서 처리).
    """
    settings = get_settings()
    if not getattr(settings, "comfyui_enabled", False):
        raise HTTPException(status_code=503, detail="ComfyUI is disabled")
    workflow_name = getattr(settings, "comfyui_ltx23_workflow", "ltx23_i2v") or "ltx23_i2v"
    workflow_path = settings.pipelines_dir / f"{workflow_name}.json"
    if not workflow_path.exists():
        workflow_path = settings.pipelines_dir / "comfyui_ltx23_workflow.json"
    if not workflow_path.exists():
        raise HTTPException(
            status_code=404,
            detail=f"Workflow not found. Tried: {workflow_name}.json and comfyui_ltx23_workflow.json in {settings.pipelines_dir}. Export from ComfyUI as API format and save to pipelines/.",
        )
    try:
        data = load_comfyui_workflow_json(workflow_path)
    except RuntimeError as e:
        raise HTTPException(status_code=500, detail=str(e))
    return data


def _comfyui_base() -> str:
    return get_settings().comfyui_base_url.rstrip("/")


@router.post("/video/comfyui/upload/image")
async def comfyui_proxy_upload_image(
    image: Annotated[UploadFile, File(description="Image file")],
):
    """ComfyUI /upload/image 프록시. 프론트가 백엔드로만 요청해도 되도록."""
    settings = get_settings()
    if not getattr(settings, "comfyui_enabled", False):
        raise HTTPException(status_code=503, detail="ComfyUI is disabled")
    content = await image.read()
    name = image.filename or "input.png"
    if not name.lower().endswith((".png", ".jpg", ".jpeg", ".webp")):
        name += ".png"
    async with httpx.AsyncClient(timeout=30.0) as client:
        r = await client.post(
            f"{_comfyui_base()}/upload/image",
            files={"image": (name, content, image.content_type or "image/png")},
            data={"type": "input", "overwrite": "true"},
        )
        r.raise_for_status()
        return r.json()


@router.post("/video/comfyui/prompt")
async def comfyui_proxy_prompt(request: Request):
    """ComfyUI /prompt 프록시. body: { prompt, prompt_id }."""
    settings = get_settings()
    if not getattr(settings, "comfyui_enabled", False):
        raise HTTPException(status_code=503, detail="ComfyUI is disabled")
    body = await request.json()
    async with httpx.AsyncClient(timeout=settings.comfyui_timeout_seconds) as client:
        r = await client.post(f"{_comfyui_base()}/prompt", json=body)
        if r.status_code >= 400:
            raise HTTPException(status_code=r.status_code, detail=r.text[:500] if r.text else "ComfyUI error")
        return r.json()


@router.get("/video/comfyui/history/{prompt_id}")
async def comfyui_proxy_history(prompt_id: str):
    """ComfyUI /history/{prompt_id} 프록시."""
    settings = get_settings()
    if not getattr(settings, "comfyui_enabled", False):
        raise HTTPException(status_code=503, detail="ComfyUI is disabled")
    async with httpx.AsyncClient(timeout=30.0) as client:
        r = await client.get(f"{_comfyui_base()}/history/{prompt_id}")
        if r.status_code >= 400:
            raise HTTPException(status_code=r.status_code, detail=r.text[:500] if r.text else "ComfyUI error")
        return r.json()


@router.get("/video/comfyui/view")
async def comfyui_proxy_view(
    filename: str,
    type: str = "output",
    subfolder: str = "",
):
    """ComfyUI /view 프록시. 비디오/이미지 바이너리 반환."""
    settings = get_settings()
    if not getattr(settings, "comfyui_enabled", False):
        raise HTTPException(status_code=503, detail="ComfyUI is disabled")
    params = {"filename": filename, "type": type}
    if subfolder:
        params["subfolder"] = subfolder
    async with httpx.AsyncClient(timeout=120.0) as client:
        r = await client.get(f"{_comfyui_base()}/view", params=params)
        if r.status_code >= 400:
            raise HTTPException(status_code=r.status_code, detail=r.text[:500] if r.text else "ComfyUI error")
        content = r.content
        ct = r.headers.get("content-type") or ""
    media = "video/mp4" if (filename.lower().endswith(".mp4") or "video" in ct) else (ct or "application/octet-stream")
    return StreamingResponse(iter([content]), media_type=media)


# ---------- Dance / Motion Transfer (Pose → Video) ----------


@router.get("/dance/motions")
async def list_dance_motions() -> list[dict[str, str]]:
    """List available dance styles for motion transfer (id, label, videoReference)."""
    return list(DANCE_MOTIONS)


@router.post("/dance/generate", response_model=DanceJobResponse)
async def generate_dance(
    background_tasks: BackgroundTasks,
    request: Request,
    motion_id: Annotated[str, Form(description="e.g. rat_dance")],
    character: Annotated[str, Form(description="dog or cat")] = "dog",
    file: Annotated[UploadFile | None, File(description="Character image (dog/cat)")] = None,
    image: Annotated[UploadFile | None, File(description="Character image (alias)")] = None,
) -> DanceJobResponse:
    """
    Generate video of character performing the selected dance (비동기 job).
    즉시 job_id 반환 → GET /dance/status/{job_id} 로 폴링.
    """
    _check_rate_limit(request)
    upload = file if (file and file.filename) else image
    if not upload or not upload.filename:
        raise HTTPException(status_code=422, detail="Missing image file. Send as 'file' or 'image'.")
    mid = (motion_id or "").strip()
    if not mid:
        raise HTTPException(status_code=400, detail="motion_id is required")
    char = (character or "dog").strip().lower()
    if char not in ("dog", "cat"):
        char = "dog"
    settings = get_settings()
    content = await upload.read()
    if len(content) == 0:
        raise HTTPException(status_code=400, detail="Empty file")
    if not upload.content_type or not upload.content_type.startswith("image/"):
        raise HTTPException(status_code=400, detail="File must be an image (e.g. image/png, image/jpeg)")
    if len(content) > settings.upload_max_bytes:
        raise HTTPException(status_code=400, detail=f"File too large. Max {settings.upload_max_size_mb}MB")
    if get_motion_video_path(mid) is None:
        raise HTTPException(
            status_code=404,
            detail=f"Motion '{mid}' not found. Add reference video to motions/ (e.g. rat_dance.mp4).",
        )

    import uuid as _uuid
    job_id = _uuid.uuid4().hex
    _dance_jobs[job_id] = {"status": "processing", "video_url": None, "processing_time": None,
                           "motion_id": mid, "character": char, "error": None}
    background_tasks.add_task(_run_dance_job, job_id, content, mid, char)
    return DanceJobResponse(job_id=job_id)


@router.post("/dance/generate-custom", response_model=DanceJobResponse)
async def generate_dance_custom(
    background_tasks: BackgroundTasks,
    request: Request,
    character: Annotated[str, Form(description="dog or cat")] = "dog",
    image: Annotated[UploadFile | None, File(description="Character image (dog/cat)")] = None,
    reference_video: Annotated[UploadFile | None, File(description="Reference video to mimic movements")] = None,
) -> DanceJobResponse:
    """
    Generate video of character mimicking movements from a user-uploaded reference video (비동기 job).
    즉시 job_id 반환 → GET /dance/status/{job_id} 로 폴링.
    """
    _check_rate_limit(request)
    if not image or not image.filename:
        raise HTTPException(status_code=422, detail="Missing character image. Send as 'image'.")
    if not reference_video or not reference_video.filename:
        raise HTTPException(status_code=422, detail="Missing reference_video file.")
    char = (character or "dog").strip().lower()
    if char not in ("dog", "cat"):
        char = "dog"
    settings = get_settings()

    image_content = await image.read()
    if len(image_content) == 0:
        raise HTTPException(status_code=400, detail="Empty image file")
    if not image.content_type or not image.content_type.startswith("image/"):
        raise HTTPException(status_code=400, detail="image must be an image file (image/png, image/jpeg, ...)")
    if len(image_content) > settings.upload_max_bytes:
        raise HTTPException(status_code=400, detail=f"Image too large. Max {settings.upload_max_size_mb}MB")

    video_content = await reference_video.read()
    if len(video_content) == 0:
        raise HTTPException(status_code=400, detail="Empty reference_video file")
    if len(video_content) > 200 * 1024 * 1024:
        raise HTTPException(status_code=400, detail="reference_video too large. Max 200MB")

    import uuid as _uuid
    job_id = _uuid.uuid4().hex
    _dance_jobs[job_id] = {"status": "processing", "video_url": None, "processing_time": None,
                           "motion_id": "custom", "character": char, "error": None}
    background_tasks.add_task(_run_dance_custom_job, job_id, image_content, video_content, char)
    return DanceJobResponse(job_id=job_id)


@router.get("/dance/status/{job_id}", response_model=DanceJobStatusResponse)
async def get_dance_job_status(job_id: str) -> DanceJobStatusResponse:
    """댄스 생성 job 상태 조회. status: processing | completed | failed"""
    job = _dance_jobs.get(job_id)
    if not job:
        raise HTTPException(status_code=404, detail="Dance job not found")
    return DanceJobStatusResponse(job_id=job_id, **job)


async def _run_dance_job(job_id: str, image_bytes: bytes, motion_id: str, character: str) -> None:
    try:
        out_bytes, processing_time = await run_dance_generate(
            image_bytes=image_bytes, motion_id=motion_id, character=character
        )
        video_filename = await save_upload_async(out_bytes, suffix=".mp4")
        video_url = get_generated_url(video_filename)
        _dance_jobs[job_id].update({"status": "completed", "video_url": video_url,
                                    "processing_time": round(processing_time, 2)})
    except Exception as e:
        logger.exception("Dance job %s failed: %s", job_id, e)
        err_msg = str(e)
        if "did not finish within" in err_msg or "TimeoutError" in type(e).__name__:
            err_msg = (
                "영상 생성이 제한 시간(약 15분) 내에 완료되지 않았습니다. "
                "ComfyUI 서버가 바쁘거나 워크플로가 지연 중일 수 있습니다. 잠시 후 다시 시도해 주세요."
            )
        _dance_jobs[job_id].update({"status": "failed", "error": err_msg})


async def _run_dance_custom_job(job_id: str, image_bytes: bytes, video_bytes: bytes, character: str) -> None:
    try:
        out_bytes, processing_time = await run_dance_generate_custom(
            image_bytes=image_bytes, reference_video_bytes=video_bytes, character=character
        )
        video_filename = await save_upload_async(out_bytes, suffix=".mp4")
        video_url = get_generated_url(video_filename)
        _dance_jobs[job_id].update({"status": "completed", "video_url": video_url,
                                    "processing_time": round(processing_time, 2)})
    except Exception as e:
        logger.exception("Dance custom job %s failed: %s", job_id, e)
        err_msg = str(e)
        # 타임아웃 시 사용자용 짧은 메시지 + 상세는 로그에
        if "did not finish within" in err_msg or "TimeoutError" in type(e).__name__:
            err_msg = (
                "영상 생성이 제한 시간(약 15분) 내에 완료되지 않았습니다. "
                "ComfyUI 서버가 바쁘거나 워크플로가 지연 중일 수 있습니다. 잠시 후 다시 시도해 주세요."
            )
        _dance_jobs[job_id].update({"status": "failed", "error": err_msg})


# ---------- LLM (gpt-oss-20b) API ----------


@router.get("/llm/status")
async def llm_status() -> dict:
    """LLM 사용 가능 여부 및 모델명 (로컬이면 모델 ID, 외부 API면 모델 이름)."""
    from app.services.llm_service import get_llm_model_display_name, is_llm_available

    return {
        "available": is_llm_available(),
        "model": get_llm_model_display_name() if is_llm_available() else None,
    }


# ---------- 채팅방 저장 ----------


def _get_chat_user_id(
    x_user_id: Annotated[str | None, Header(alias="X-User-Id")] = None,
    body: dict | None = None,
) -> str:
    """채팅 API용 user_id. 헤더 X-User-Id 또는 body user_id. 없으면 400."""
    user_id = (x_user_id or "").strip() or ((body or {}).get("user_id") or "").strip()
    if not user_id:
        raise HTTPException(status_code=400, detail="X-User-Id header or user_id in body required")
    return user_id


@router.get("/chat/rooms")
async def chat_list_rooms(
    x_user_id: Annotated[str | None, Header(alias="X-User-Id")] = None,
) -> list[dict]:
    """저장된 채팅방 목록 (user_id 기준, 최신순)."""
    from app.services.chat_store import list_rooms
    user_id = _get_chat_user_id(x_user_id=x_user_id)
    return list_rooms(user_id)


@router.get("/chat/rooms/{room_id}")
async def chat_get_room(
    room_id: str,
    x_user_id: Annotated[str | None, Header(alias="X-User-Id")] = None,
) -> dict:
    """채팅방 한 건 조회 (user_id 소유만)."""
    from app.services.chat_store import get_room
    user_id = _get_chat_user_id(x_user_id=x_user_id)
    room = get_room(room_id, user_id)
    if room is None:
        raise HTTPException(status_code=404, detail="Room not found")
    return room


@router.post("/chat/rooms")
async def chat_create_room(
    body: Annotated[dict, Body()] = None,
    x_user_id: Annotated[str | None, Header(alias="X-User-Id")] = None,
) -> dict:
    """채팅방 생성. body: { title?: string, user_id?: string }. user_id 없으면 X-User-Id 헤더 필수."""
    from app.services.chat_store import create_room
    user_id = _get_chat_user_id(x_user_id=x_user_id, body=body or {})
    title = (body or {}).get("title") if body else None
    if isinstance(title, str):
        title = title.strip() or None
    return create_room(user_id, title=title)


@router.post("/chat/rooms/{room_id}/messages")
async def chat_add_message(
    room_id: str,
    body: Annotated[dict, Body()],
    x_user_id: Annotated[str | None, Header(alias="X-User-Id")] = None,
) -> dict:
    """채팅방에 메시지 추가. body: { role, content [, user_id] }."""
    from app.services.chat_store import add_message
    user_id = _get_chat_user_id(x_user_id=x_user_id, body=body or {})
    role = (body or {}).get("role", "user")
    content = (body or {}).get("content", "")
    if role not in ("user", "assistant"):
        raise HTTPException(status_code=400, detail="role must be user or assistant")
    updated = add_message(room_id, user_id, role, str(content))
    if updated is None:
        raise HTTPException(status_code=404, detail="Room not found")
    return updated


@router.delete("/chat/rooms/{room_id}")
async def chat_delete_room(
    room_id: str,
    x_user_id: Annotated[str | None, Header(alias="X-User-Id")] = None,
) -> dict:
    """채팅방 삭제 (user_id 소유만)."""
    from app.services.chat_store import delete_room
    user_id = _get_chat_user_id(x_user_id=x_user_id)
    if not delete_room(room_id, user_id):
        raise HTTPException(status_code=404, detail="Room not found")
    return {"ok": True}


@router.post("/llm/chat")
async def llm_chat(
    request: Request,
    body: Annotated[dict, Body()],
) -> dict:
    """건강 질문 도우미 채팅. 응답에 1~5분 걸릴 수 있음. body: { messages, max_tokens?, temperature? }"""
    _check_rate_limit(request)
    from app.services.llm_service import is_llm_available, complete_health_chat, LLMQueueTimeoutError

    if not is_llm_available():
        raise HTTPException(status_code=503, detail="LLM not available")
    messages = body.get("messages") or []
    if not messages:
        raise HTTPException(status_code=400, detail="messages required")
    max_tokens = int(body.get("max_tokens", 2048) or 2048)
    temperature = float(body.get("temperature", 0.4) or 0.4)
    logger.info("LLM chat request: %s messages, max_tokens=%s", len(messages), max_tokens)
    try:
        text, structured = await complete_health_chat(messages, max_tokens=max_tokens, temperature=temperature)
    except LLMQueueTimeoutError as e:
        raise HTTPException(status_code=503, detail=str(e))
    except Exception as e:
        logger.exception("LLM chat error: %s", e)
        raise HTTPException(status_code=503, detail="LLM request failed") from e
    if text is None:
        logger.warning("LLM chat returned None")
        raise HTTPException(status_code=503, detail="LLM request failed")
    if not text.strip():
        text = "응답을 생성하지 못했습니다. 잠시 후 다시 시도해 주세요."
    logger.info("LLM chat response length: %s chars, structured: %s", len(text), structured is not None)
    out: dict = {"content": text}
    if structured is not None:
        out["structured"] = structured.model_dump()
    return out


async def _stream_llm_chat_body(
    messages: list,
    max_tokens: int,
    temperature: float,
):
    """스트리밍: 한 줄에 하나씩 JSON { \"content\": \"chunk\" } 전송."""
    from app.services.llm_service import stream_complete_health_chat

    async for chunk in stream_complete_health_chat(messages, max_tokens=max_tokens, temperature=temperature):
        yield _json.dumps({"content": chunk}, ensure_ascii=False) + "\n"


@router.post("/llm/chat/stream")
async def llm_chat_stream(
    request: Request,
    body: Annotated[dict, Body()],
):
    """건강 질문 채팅 스트리밍. 응답은 NDJSON(한 줄당 {\"content\": \"chunk\"})."""
    _check_rate_limit(request)
    from app.services.llm_service import is_llm_available

    if not is_llm_available():
        raise HTTPException(status_code=503, detail="LLM not available")
    messages = body.get("messages") or []
    if not messages:
        raise HTTPException(status_code=400, detail="messages required")
    max_tokens = int(body.get("max_tokens", 2048) or 2048)
    temperature = float(body.get("temperature", 0.4) or 0.4)
    # 프록시가 스트리밍 응답을 버퍼링하면 청크가 끊겨 ERR_INCOMPLETE_CHUNKED_ENCODING 발생.
    # nginx: X-Accel-Buffering: no 로 버퍼링 비활성화.
    headers = {
        "Cache-Control": "no-cache, no-store",
        "X-Accel-Buffering": "no",
        "Connection": "keep-alive",
    }
    return StreamingResponse(
        _stream_llm_chat_body(messages, max_tokens=max_tokens, temperature=temperature),
        media_type="application/x-ndjson",
        headers=headers,
    )


@router.post("/llm/suggest-prompt")
async def llm_suggest_prompt(
    request: Request,
    body: Annotated[dict, Body()],
) -> dict:
    """이미지 생성용 프롬프트 추천. body: { style: string, user_hint?: string }"""
    _check_rate_limit(request)
    from app.services.llm_service import is_llm_available, suggest_prompt_for_style

    if not is_llm_available():
        raise HTTPException(status_code=503, detail="LLM not available")
    style = (body.get("style") or "").strip() or "realistic"
    user_hint = (body.get("user_hint") or "").strip() or None
    prompt = await suggest_prompt_for_style(style, user_hint)
    if prompt is None:
        raise HTTPException(status_code=503, detail="LLM request failed")
    return {"prompt": prompt}


# ---------- LoRA 학습 데이터 API ----------


def _training_item_with_url(item: dict) -> dict:
    """항목에 image_url 추가."""
    base = "/api/training/images"
    return {**item, "image_url": f"{base}/{item['image_filename']}"}


@router.get("/training/items")
async def training_list(
    category: str | None = None,
) -> list[dict]:
    """학습용 데이터 목록 (이미지 URL 포함). category 쿼리로 필터 가능."""
    items = training_list_items(category=category)
    return [_training_item_with_url(it) for it in items]


@router.get("/training/categories")
async def training_categories() -> list[str]:
    """학습 데이터에 사용 중인 카테고리 목록."""
    return training_list_categories()


@router.post("/training/items")
async def training_add(
    request: Request,
    file: Annotated[UploadFile, File(description="학습용 이미지")],
    caption: Annotated[str, Form(description="프롬프트 라벨")] = "",
    category: Annotated[str, Form(description="카테고리 (예: 픽셀아트, anime)")] = "",
) -> dict:
    """학습 데이터 1건 추가 (이미지 + 캡션 + 카테고리)."""
    if not file.content_type or not file.content_type.startswith("image/"):
        raise HTTPException(status_code=400, detail="Image file required")
    content = await file.read()
    if len(content) == 0:
        raise HTTPException(status_code=400, detail="Empty file")
    settings = get_settings()
    if len(content) > settings.upload_max_bytes:
        raise HTTPException(status_code=400, detail=f"File too large. Max {settings.upload_max_size_mb}MB")
    item = training_add_item(content, caption, category=category)
    return _training_item_with_url(item)


@router.patch("/training/items/{item_id}")
async def training_update_item_route(
    item_id: str,
    body: Annotated[dict, Body()] = None,
) -> dict | None:
    """학습 항목의 캡션·카테고리 수정. body: { \"caption\": \"...\", \"category\": \"...\" }"""
    body = body or {}
    caption = body.get("caption") if "caption" in body else None
    category = body.get("category") if "category" in body else None
    if caption is None and category is None:
        raise HTTPException(status_code=400, detail="caption or category required")
    updated = training_update_item(item_id, caption=caption, category=category)
    if updated is None:
        raise HTTPException(status_code=404, detail="Item not found")
    return _training_item_with_url(updated)


@router.delete("/training/items/{item_id}")
async def training_delete(item_id: str) -> dict:
    """학습 데이터 1건 삭제."""
    if not training_delete_item(item_id):
        raise HTTPException(status_code=404, detail="Item not found")
    return {"ok": True}


@router.get("/training/images/{filename:path}")
async def training_serve_image(filename: str) -> FileResponse:
    """학습용 이미지 파일 서빙."""
    path = training_get_image_path(filename)
    if path is None or not path.is_file():
        raise HTTPException(status_code=404, detail="Image not found")
    return FileResponse(path, media_type="image/png")


@router.post("/training/start")
async def training_start(
    body: Annotated[dict, Body()] = None,
) -> dict:
    """LoRA 학습 시작 (백그라운드). body: { \"category\": \"픽셀아트\" } — 해당 카테고리만 학습. 생략 시 전체."""
    from app.services.training_runner import start_lora_training

    category = (body or {}).get("category") if body else None
    if isinstance(category, str) and not category.strip():
        category = None
    result = start_lora_training(category=category)
    if result.get("error"):
        raise HTTPException(status_code=400, detail=result["error"])
    return result


# ---------- ComfyUI (로컬 ComfyUI 서버 연동) ----------


@router.get("/comfyui/health")
async def comfyui_health():
    """ComfyUI 서버 연결 여부."""
    settings = get_settings()
    if not settings.comfyui_enabled:
        return {"enabled": False, "connected": False}
    connected = await comfyui_health_check()
    return {"enabled": True, "connected": connected, "base_url": settings.comfyui_base_url}


@router.get("/comfyui/pipelines")
async def comfyui_list_pipelines():
    """pipelines/ 디렉터리 내 .json 워크플로우 목록 (docs 경로 그대로)."""
    settings = get_settings()
    pipelines_dir = settings.pipelines_dir
    if not pipelines_dir.exists():
        return {"pipelines": []}
    names = [p.stem for p in pipelines_dir.glob("*.json")]
    return {"pipelines": sorted(names)}


@router.post("/comfyui/run", response_model=ComfyUIRunResponse)
async def comfyui_run(body: ComfyUIRunRequest):
    """
    ComfyUI 워크플로우 실행. pipeline_name 이 있으면 pipelines/{name}.json 로드 후 실행.
    workflow 직접 전달도 가능. 완료 시 이미지 bytes 반환 또는 static/generated 에 저장 후 URL 반환.
    """
    settings = get_settings()
    if not settings.comfyui_enabled:
        return ComfyUIRunResponse(success=False, error="ComfyUI is disabled")
    workflow = body.workflow
    if body.pipeline_name:
        path = settings.pipelines_dir / f"{body.pipeline_name.replace('.json', '')}.json"
        if not path.exists():
            return ComfyUIRunResponse(success=False, error=f"Pipeline not found: {path.name}")
        workflow = _json.loads(path.read_text(encoding="utf-8"))
    if not workflow:
        return ComfyUIRunResponse(success=False, error="Empty workflow")
    try:
        out_path = await run_workflow_save_to_generated(workflow, output_name=None)
        filename = out_path.name
        url = get_generated_url(filename)
        return ComfyUIRunResponse(success=True, image_url=url)
    except TimeoutError as e:
        return ComfyUIRunResponse(success=False, error=str(e))
    except RuntimeError as e:
        return ComfyUIRunResponse(success=False, error=str(e))
    except Exception as e:
        logger.exception("ComfyUI run failed")
        return ComfyUIRunResponse(success=False, error=str(e))
