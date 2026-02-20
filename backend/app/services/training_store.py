"""
LoRA 학습용 데이터 저장소: 이미지 파일 + 캡션(프롬프트) JSON.
"""

from __future__ import annotations

import json
import logging
import uuid
from pathlib import Path
from typing import Any

from app.core.config import get_settings

logger = logging.getLogger(__name__)

METADATA_FILENAME = "dataset.json"
IMAGES_SUBDIR = "images"


def _get_training_dir() -> Path:
    settings = get_settings()
    path = settings.training_dir
    path.mkdir(parents=True, exist_ok=True)
    return path


def _get_images_dir() -> Path:
    d = _get_training_dir() / IMAGES_SUBDIR
    d.mkdir(parents=True, exist_ok=True)
    return d


def _get_metadata_path() -> Path:
    return _get_training_dir() / METADATA_FILENAME


def _load_metadata() -> list[dict[str, Any]]:
    path = _get_metadata_path()
    if not path.exists():
        return []
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
        return data if isinstance(data, list) else []
    except Exception as e:
        logger.warning("Failed to load training metadata: %s", e)
        return []


def _save_metadata(items: list[dict[str, Any]]) -> None:
    _get_metadata_path().write_text(
        json.dumps(items, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )


def list_items() -> list[dict[str, Any]]:
    """학습 데이터 목록. 각 항목: id, image_filename, caption, created_at."""
    return _load_metadata()


def add_item(image_bytes: bytes, caption: str) -> dict[str, Any]:
    """이미지와 캡션을 저장하고 항목 정보 반환."""
    items = _load_metadata()
    images_dir = _get_images_dir()
    ext = ".png"
    name = f"{uuid.uuid4().hex}{ext}"
    path = images_dir / name
    path.write_bytes(image_bytes)
    item = {
        "id": uuid.uuid4().hex,
        "image_filename": name,
        "caption": (caption or "").strip(),
        "created_at": __import__("datetime").datetime.utcnow().isoformat() + "Z",
    }
    items.append(item)
    _save_metadata(items)
    return item


def update_caption(item_id: str, caption: str) -> dict[str, Any] | None:
    """캡션만 수정. 항목 없으면 None."""
    items = _load_metadata()
    for i, it in enumerate(items):
        if it.get("id") == item_id:
            items[i] = {**it, "caption": (caption or "").strip()}
            _save_metadata(items)
            return items[i]
    return None


def delete_item(item_id: str) -> bool:
    """항목 및 해당 이미지 파일 삭제. 없으면 False."""
    items = _load_metadata()
    images_dir = _get_images_dir()
    for i, it in enumerate(items):
        if it.get("id") == item_id:
            fn = it.get("image_filename")
            if fn:
                (images_dir / fn).unlink(missing_ok=True)
            items.pop(i)
            _save_metadata(items)
            return True
    return False


def get_image_path(image_filename: str) -> Path | None:
    """이미지 파일의 절대 경로. 없으면 None."""
    p = _get_images_dir() / image_filename
    return p if p.exists() else None
