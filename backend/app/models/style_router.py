"""
Style-based model and scheduler routing. Extensible.
"""

from __future__ import annotations

from typing import Final

ModelKey = str

SDXL_BASE: Final[ModelKey] = "sdxl_base"
ANIMAGINE_XL: Final[ModelKey] = "animagine_xl"

STYLE_TO_MODEL: Final[dict[str, ModelKey]] = {
    "realistic": SDXL_BASE,
    "cinematic": SDXL_BASE,
    "fantasy art": SDXL_BASE,
    "3d render": SDXL_BASE,
    "watercolor": SDXL_BASE,
    "oil painting": SDXL_BASE,
    "sketch": SDXL_BASE,
    "cyberpunk": SDXL_BASE,
    "anime": ANIMAGINE_XL,
    "pixel art": SDXL_BASE,
}

DEFAULT_MODEL: Final[ModelKey] = SDXL_BASE


def get_model_key_for_style(style: str) -> ModelKey:
    key = (style or "").strip().lower()
    return STYLE_TO_MODEL.get(key, DEFAULT_MODEL)


def list_styles() -> list[str]:
    return sorted(set(STYLE_TO_MODEL.keys()))
