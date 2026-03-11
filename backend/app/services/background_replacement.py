"""
Background Override System for cloud-themed pet generation.

구조: Subject 분리 → 배경만 생성 → 합성. Subject 영역에는 노이즈/재생성 없음 (identity 유지).

1. Segment pet (rembg) → subject 픽셀만 추출.
2. 배경만 구름 생성 (solid → img2img). Subject 미포함.
3. 합성: original subject + generated cloud. Subject 픽셀 수정 없음. 전체 프레임 img2img 없음.

(설계·Turbo 한계·inpaint/QA는 코드 주석 및 API 문서 참고)
"""

from __future__ import annotations

import io
import logging
from typing import Any

logger = logging.getLogger(__name__)

# Optional: rembg for segmentation. If not installed, segment_foreground returns None.
try:
    from rembg import remove as rembg_remove
    _REMBG_AVAILABLE = True
except ImportError:
    _REMBG_AVAILABLE = False
    rembg_remove = None  # type: ignore[misc, assignment]


def is_background_replacement_available() -> bool:
    """True if segmentation (rembg) is available for programmatic background replacement."""
    return _REMBG_AVAILABLE


def segment_foreground(image_bytes: bytes) -> bytes | None:
    """
    Step 1: Detect and segment the pet from the image.
    Returns RGBA PNG bytes (pet with transparent background), or None if segmentation unavailable/failed.
    """
    if not _REMBG_AVAILABLE:
        logger.warning("rembg not installed; background replacement unavailable. pip install rembg[cpu]")
        return None
    try:
        from PIL import Image, ImageOps
        img = Image.open(io.BytesIO(image_bytes))
        img = ImageOps.exif_transpose(img)
        img = img.convert("RGB")
        out = rembg_remove(img)
        if out is None:
            return None
        out = out.convert("RGBA")
        buf = io.BytesIO()
        out.save(buf, format="PNG")
        return buf.getvalue()
    except Exception as e:
        logger.exception("Segment foreground failed: %s", e)
        return None


def composite_foreground_on_background(
    foreground_rgba_bytes: bytes,
    background_rgb_pil: Any,
    align: str = "center",
) -> bytes:
    """
    Step 4 & 5: Place foreground (pet with alpha) on top of background; blend.
    Sizes must match. Returns RGB PNG bytes.
    """
    from PIL import Image
    fg = Image.open(io.BytesIO(foreground_rgba_bytes))
    fg = fg.convert("RGBA")
    bg = background_rgb_pil.convert("RGB")
    if fg.size != bg.size:
        bg = bg.resize(fg.size, Image.Resampling.LANCZOS)
    bg_rgba = Image.new("RGBA", fg.size)
    bg_rgba.paste(bg, (0, 0))
    composite = Image.alpha_composite(bg_rgba, fg)
    out = composite.convert("RGB")
    buf = io.BytesIO()
    out.save(buf, format="PNG")
    return buf.getvalue()


def create_cloud_background_prompt(photoreal: bool = False) -> str:
    """Prompt for generating cloud-only background (no pet, no ground)."""
    if photoreal:
        return (
            "Bright white clouds, soft sky blue atmosphere, realistic sky, "
            "natural clouds, overcast daylight. No ground. No indoor. No animals. No people. Only sky and clouds."
        )
    return (
        "Bright white volumetric clouds, soft sky blue atmosphere, "
        "large soft luminous clouds, overcast sky. No ground. No indoor. No animals. No people. Only clouds and sky."
    )


def create_cloud_background_negative() -> str:
    """Negative for cloud background generation."""
    return (
        "dark sky, night, sunset, ground, floor, indoor, furniture, animals, people, "
        "trees, grass, buildings, dramatic lighting, low contrast"
    )
