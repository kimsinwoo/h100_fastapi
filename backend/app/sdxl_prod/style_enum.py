"""
Style enum: single source of truth. No string comparison; enum-based mapping only.
Invalid style → ValueError (no fallback).
"""
from __future__ import annotations

from enum import Enum


class Style(str, Enum):
    ANIME = "anime"
    REALISTIC = "realistic"
    WATERCOLOR = "watercolor"
    CYBERPUNK = "cyberpunk"
    OIL = "oil_painting"
    SKETCH = "sketch"
    CINEMATIC = "cinematic"
    FANTASY = "fantasy_art"
    PIXEL = "pixel_art"
    RENDER3D = "3d_render"


def style_from_request_value(value: str) -> Style:
    """Raise ValueError if value is not a valid Style. No fallback."""
    try:
        return Style(value)
    except ValueError:
        raise ValueError(f"Invalid style: {value!r}") from None
