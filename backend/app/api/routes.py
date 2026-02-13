"""
Image generation API routes.
"""

from __future__ import annotations

import logging
from typing import Annotated

from fastapi import APIRouter, File, Form, HTTPException, Request, UploadFile

from app.core.config import get_settings
from app.models.style_presets import STYLE_PRESETS
from app.schemas.image_schema import GenerateResponse
from app.services.image_service import run_image_to_image
from app.utils.file_handler import get_generated_url, save_upload_async

logger = logging.getLogger(__name__)

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


@router.post("/generate", response_model=GenerateResponse)
async def generate(
    request: Request,
    file: Annotated[UploadFile, File(description="Image file to transform")],
    style: Annotated[str, Form(description="Style preset key")] = "realistic",
    custom_prompt: Annotated[str | None, Form(description="Optional custom prompt")] = None,
    strength: Annotated[float | None, Form()] = None,
    seed: Annotated[int | None, Form()] = None,
) -> GenerateResponse:
    """
    Upload image, run Z-Image-Turbo with selected style (and optional custom prompt).
    Returns URLs to original and generated images plus processing time.
    """
    _check_rate_limit(request)
    settings = get_settings()
    style_lower = style.strip().lower()
    if style_lower not in STYLE_PRESETS:
        raise HTTPException(
            status_code=400,
            detail=f"Invalid style. Allowed: {', '.join(sorted(STYLE_PRESETS.keys()))}",
        )
    if not file.content_type or not file.content_type.startswith("image/"):
        raise HTTPException(status_code=400, detail="File must be an image (e.g. image/png, image/jpeg)")
    content = await file.read()
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
    try:
        out_bytes, processing_time = await run_image_to_image(
            image_bytes=content,
            style_key=style_lower,
            custom_prompt=custom_prompt,
            strength=strength,
            seed=seed,
        )
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except RuntimeError as e:
        logger.exception("Model inference failed: %s", e)
        raise HTTPException(status_code=503, detail=str(e))
    try:
        generated_filename = await save_upload_async(out_bytes, suffix=".png")
    except Exception as e:
        logger.exception("Failed to save generated image: %s", e)
        raise HTTPException(status_code=500, detail="Failed to save generated image")
    generated_url = get_generated_url(generated_filename)
    return GenerateResponse(
        original_url=original_url,
        generated_url=generated_url,
        processing_time=round(processing_time, 2),
    )


@router.get("/styles")
async def list_styles() -> dict[str, str]:
    """Return available style presets (key -> description)."""
    return {k: v for k, v in STYLE_PRESETS.items()}
