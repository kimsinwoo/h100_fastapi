"""
Image generation API routes.
"""

from __future__ import annotations

import logging
import sys
from typing import Annotated

from fastapi import APIRouter, Body, File, Form, HTTPException, Request, UploadFile
from fastapi.responses import FileResponse

from app.core.config import get_settings
from app.models.style_presets import STYLE_PRESETS
from app.schemas.image_schema import GenerateResponse
from app.services.image_service import run_image_to_image
from app.services.training_store import (
    add_item as training_add_item,
    delete_item as training_delete_item,
    get_image_path as training_get_image_path,
    list_items as training_list_items,
    update_caption as training_update_caption,
)
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
    print("[Z-Image] 로컬에서 이미지 생성 중...", file=sys.stderr, flush=True)
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
    print("[Z-Image] 로컬 생성 완료 (%.1f초)" % processing_time, file=sys.stderr, flush=True)
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


# ---------- LoRA 학습 데이터 API ----------


def _training_item_with_url(item: dict) -> dict:
    """항목에 image_url 추가."""
    base = "/api/training/images"
    return {**item, "image_url": f"{base}/{item['image_filename']}"}


@router.get("/training/items")
async def training_list() -> list[dict]:
    """학습용 데이터 목록 (이미지 URL 포함)."""
    items = training_list_items()
    return [_training_item_with_url(it) for it in items]


@router.post("/training/items")
async def training_add(
    request: Request,
    file: Annotated[UploadFile, File(description="학습용 이미지")],
    caption: Annotated[str, Form(description="프롬프트 라벨")] = "",
) -> dict:
    """학습 데이터 1건 추가 (이미지 + 캡션)."""
    if not file.content_type or not file.content_type.startswith("image/"):
        raise HTTPException(status_code=400, detail="Image file required")
    content = await file.read()
    if len(content) == 0:
        raise HTTPException(status_code=400, detail="Empty file")
    settings = get_settings()
    if len(content) > settings.upload_max_bytes:
        raise HTTPException(status_code=400, detail=f"File too large. Max {settings.upload_max_size_mb}MB")
    item = training_add_item(content, caption)
    return _training_item_with_url(item)


@router.patch("/training/items/{item_id}")
async def training_update_caption_route(
    item_id: str,
    body: Annotated[dict, Body()] = None,
) -> dict | None:
    """학습 항목의 캡션(프롬프트)만 수정. body: { \"caption\": \"...\" }"""
    caption = (body or {}).get("caption", "")
    updated = training_update_caption(item_id, caption)
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
async def training_start() -> dict:
    """LoRA 학습 시작 (백그라운드). 데이터가 없으면 400."""
    from app.services.training_runner import start_lora_training

    result = start_lora_training()
    if result.get("error"):
        raise HTTPException(status_code=400, detail=result["error"])
    return result
