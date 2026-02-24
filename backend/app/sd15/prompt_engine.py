"""
Strict Prompt Enforcement. final_prompt = STYLE_PREFIX + user_prompt + QUALITY_SUFFIX.
Style-specific negative. User prompt never overrides style.
"""
from __future__ import annotations

from app.sd15.schemas import StyleKind

REALISTIC_STYLE_PREFIX: str = (
    "RAW professional photograph, "
    "natural skin texture, "
    "true-to-life color grading, "
    "cinematic lighting, "
    "high dynamic range, "
    "subtle film grain, "
    "natural imperfections, "
    "realistic lens behavior, "
)
REALISTIC_QUALITY_SUFFIX: str = (
    "ultra detailed, "
    "sharp focus, "
    "high resolution, "
    "fine texture detail, "
    "real skin pores, "
    "balanced exposure"
)
REALISTIC_NEGATIVE: str = (
    "cartoon, anime, illustration, painting, cgi, 3d render, "
    "plastic skin, smooth skin, oversharpened, unrealistic lighting, "
    "low quality, blurry, overprocessed, exaggerated features"
)

ANIME_STYLE_PREFIX: str = (
    "STRICT 2D Japanese anime frame, "
    "clean lineart, cel shading, flat color shading, "
    "anime lighting, vibrant but controlled colors, "
)
ANIME_QUALITY_SUFFIX: str = (
    "high detail lineart, crisp edges, "
    "studio quality anime frame, sharp outlines"
)
ANIME_NEGATIVE: str = (
    "photo, realistic, 3d render, skin texture, pore detail, "
    "film grain, cinematic photography, lens blur"
)

STYLE_PREFIX: dict[str, str] = {"realistic": REALISTIC_STYLE_PREFIX, "anime": ANIME_STYLE_PREFIX}
QUALITY_SUFFIX: dict[str, str] = {"realistic": REALISTIC_QUALITY_SUFFIX, "anime": ANIME_QUALITY_SUFFIX}
NEGATIVE_PROMPT: dict[str, str] = {"realistic": REALISTIC_NEGATIVE, "anime": ANIME_NEGATIVE}
DEFAULT_STYLE: StyleKind = "realistic"


def build_prompt(user_prompt: str, style: StyleKind) -> str:
    prefix = STYLE_PREFIX.get(style, STYLE_PREFIX[DEFAULT_STYLE])
    suffix = QUALITY_SUFFIX.get(style, QUALITY_SUFFIX[DEFAULT_STYLE])
    return prefix + user_prompt.strip() + ", " + suffix


def get_negative_prompt(style: StyleKind) -> str:
    return NEGATIVE_PROMPT.get(style, NEGATIVE_PROMPT[DEFAULT_STYLE])
