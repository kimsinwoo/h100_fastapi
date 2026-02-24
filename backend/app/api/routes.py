"""
Image generation API routes.
"""

from __future__ import annotations

import logging
import sys
from typing import Annotated

from fastapi import APIRouter, Body, File, Form, Header, HTTPException, Request, UploadFile
from fastapi.responses import FileResponse

from app.core.config import get_settings
from app.models.image_prompt_expert import ImagePromptExpert
from app.models.style_presets import STYLE_PRESETS
from app.schemas.image_schema import GenerateResponse
from app.services.image_service import run_image_to_image
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
    style: Annotated[str, Form(description="Style preset key")] = "realistic",
    custom_prompt: Annotated[str | None, Form(description="Optional custom prompt")] = None,
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
    settings = get_settings()
    style_lower = style.strip().lower()
    allowed = set(ImagePromptExpert.get_allowed_style_keys())
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
    print("[Z-Image] 로컬에서 이미지 생성 중...", file=sys.stderr, flush=True)
    try:
        out_bytes, processing_time = await run_image_to_image(
            image_bytes=content,
            style_key=style_lower,
            custom_prompt=custom_prompt,
            strength=strength_f,
            seed=seed_i,
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
    """Return available style presets (key -> description). ImagePromptExpert와 동기화된 키만 노출."""
    allowed = ImagePromptExpert.get_allowed_style_keys()
    return {k: STYLE_PRESETS.get(k, k) for k in allowed}


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
    max_tokens = int(body.get("max_tokens", 1024) or 1024)
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
