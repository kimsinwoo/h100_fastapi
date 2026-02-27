"""
Clean prompt builder. No base prompt injection.
User prompt used as-is unless a style is explicitly selected.
"""

from __future__ import annotations

# Minimal style suffix only. No prepended base, no camera/lens/metadata.
STYLE_TEMPLATES: dict[str, str] = {
    "photoreal": "Ultra photorealistic, natural lighting, realistic textures",
    "realistic": "Ultra photorealistic, natural lighting, realistic textures",
    "pixel art": "Pixel art style, 16-bit, square pixels, limited color palette, no anti-aliasing",
    "pixel_art": "Pixel art style, 16-bit, square pixels, limited color palette, no anti-aliasing",
    "oil painting": "Oil painting, visible brush strokes, canvas texture",
    "anime": "Anime style, cel shading, clean outlines, vibrant colors",
    "watercolor": "Watercolor painting, soft edges, paper texture",
    "cyberpunk": "Cyberpunk, neon lights, high contrast, futuristic",
    "sketch": "Pencil sketch, monochrome, hand-drawn",
    "cinematic": "Cinematic lighting, film look, dramatic composition",
    "fantasy art": "Fantasy illustration, detailed, imaginative",
    "3d render": "3D render, CGI, clean lighting",
    "3d_render": "3D render, CGI, clean lighting",
    "omni": "High detail, sharp focus, natural colors",
}

# Negative only when style is set. No hidden defaults.
NEGATIVE_BY_STYLE: dict[str, str] = {
    "pixel art": "realistic, photorealistic, smooth shading, HDR",
    "pixel_art": "realistic, photorealistic, smooth shading, HDR",
    "photoreal": "cartoon, anime, pixel art",
    "realistic": "cartoon, anime, pixel art",
    "anime": "photorealistic, 3D render, realistic",
    "watercolor": "digital, photograph, 3D",
    "oil painting": "digital, photograph, flat color",
    "cyberpunk": "daylight, natural, pastoral",
    "sketch": "color, photograph, 3D",
    "cinematic": "cartoon, flat lighting",
    "fantasy art": "photograph, modern, minimalist",
    "3d render": "2D, flat illustration, photograph",
    "3d_render": "2D, flat illustration, photograph",
    "omni": "blurry, distorted, text, watermark",
}

ALLOWED_STYLE_KEYS = [
    "anime",
    "realistic",
    "watercolor",
    "cyberpunk",
    "oil painting",
    "sketch",
    "cinematic",
    "fantasy art",
    "pixel art",
    "3d render",
    "omni",
]


def _normalize_style(style: str | None) -> str | None:
    if style is None or not str(style).strip():
        return None
    key = str(style).strip().lower()
    key_no_space = key.replace(" ", "_")
    if key in STYLE_TEMPLATES:
        return key
    if key_no_space in STYLE_TEMPLATES:
        return key_no_space
    for k in STYLE_TEMPLATES:
        if k.replace(" ", "_") == key_no_space:
            return k
    return key


def build_prompt(user_prompt: str, style: str | None = None, raw_prompt: bool = False) -> str:
    """
    Build final prompt. No base injection.
    - If user_prompt is empty → raise ValueError.
    - If raw_prompt is True → return user_prompt unchanged.
    - If style is None → return user_prompt unchanged.
    - If style is provided → return user_prompt + minimal style template only.
    """
    text = (user_prompt or "").strip()
    if not text:
        raise ValueError("Prompt cannot be empty")
    if raw_prompt:
        return text
    style_key = _normalize_style(style)
    if style_key is None:
        return text
    template = STYLE_TEMPLATES.get(style_key)
    if not template:
        return text
    return f"{text}, {template}"


def build_negative_prompt(style: str | None, raw_prompt: bool = False) -> str:
    """
    Build negative prompt. No hidden defaults.
    - If raw_prompt or style is None → return empty string.
    - Otherwise return only the style-specific negative (minimal).
    """
    if raw_prompt:
        return ""
    style_key = _normalize_style(style)
    if style_key is None:
        return ""
    return NEGATIVE_BY_STYLE.get(style_key, "")


def get_allowed_style_keys() -> list[str]:
    """API validation: list of allowed style keys."""
    return list(ALLOWED_STYLE_KEYS)
