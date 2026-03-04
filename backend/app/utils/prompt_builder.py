"""
Clean prompt builder.
공통 전제(무조건 포함): 작은 흰 개 기반 2D 캐릭터, 실사 차단 Negative.
"""

from __future__ import annotations

# ========== 공통 전제 (무조건 포함) ==========
# ✅ 공통 Positive 베이스: 2D 캐릭터 재해석, 실사/털 텍스처 제거
COMMON_POSITIVE_BASE = (
    "fully redesigned 2D character based on a small white dog, "
    "shape-based character design, no biological realism, no fur texture, "
    "simplified anatomy, clear silhouette, flat color regions, "
    "style-locked rendering, not a real dog, animated character reinterpretation"
)
# ❌ 공통 Negative (실사 차단)
COMMON_NEGATIVE = (
    "photorealistic fur, wet nose texture, animal photography, hyper detailed hair, "
    "realistic lighting, subsurface scattering, skin pores, 3D render, "
    "cinematic HDR, depth of field, real dog anatomy"
)

# Minimal style suffix only.
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
    Build final prompt. 공통 Positive 베이스는 style 사용 시 항상 포함.
    - If user_prompt is empty → raise ValueError.
    - If raw_prompt is True → return user_prompt unchanged (공통 미포함).
    - If style is None → return user_prompt + COMMON_POSITIVE_BASE.
    - If style is provided → return user_prompt + style template + COMMON_POSITIVE_BASE.
    """
    text = (user_prompt or "").strip()
    if not text:
        raise ValueError("Prompt cannot be empty")
    if raw_prompt:
        return text
    style_key = _normalize_style(style)
    if style_key is None:
        return f"{text}, {COMMON_POSITIVE_BASE}"
    template = STYLE_TEMPLATES.get(style_key)
    if not template:
        return f"{text}, {COMMON_POSITIVE_BASE}"
    return f"{text}, {template}, {COMMON_POSITIVE_BASE}"


def build_negative_prompt(style: str | None, raw_prompt: bool = False) -> str:
    """
    Build negative prompt. 공통 Negative(실사 차단)는 항상 앞에 포함.
    - If raw_prompt → return empty string.
    - Otherwise return COMMON_NEGATIVE + style-specific negative.
    """
    if raw_prompt:
        return ""
    style_key = _normalize_style(style)
    style_neg = NEGATIVE_BY_STYLE.get(style_key, "") if style_key else ""
    if style_neg:
        return f"{COMMON_NEGATIVE}, {style_neg}"
    return COMMON_NEGATIVE


def get_allowed_style_keys() -> list[str]:
    """API validation: list of allowed style keys."""
    return list(ALLOWED_STYLE_KEYS)
