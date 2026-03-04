"""
2D character redesign prompt system for Z-Image-Turbo (H100).
Strict non-photorealistic styles; construction rules per style.
Never describe a real dog — always a redesigned animated character based on a small white dog.
"""

from __future__ import annotations

from typing import Any

# ========== BASE (always applied) ==========
# Forces 2D character redesign; removes biological realism, fur, photo lighting, 3D.
BASE_PROMPT = (
    "redesigned 2D animated character based on a small white dog, "
    "shape-based character design, not a real dog, "
    "no biological realism, no fur texture, no photographic lighting, no 3D rendering, "
    "simplified anatomy, clear silhouette, flat color regions, style-locked rendering"
)

# Blocks photorealism, realistic fur, cinematic lighting, SSS, DOF, HDR, real anatomy.
BASE_NEGATIVE = (
    "photorealistic, realistic fur, wet nose, animal photography, hyper detailed hair, "
    "cinematic lighting, subsurface scattering, depth of field, HDR, "
    "real dog anatomy, skin pores, 3D render, biological realism"
)

# ========== STYLE PROMPTS (visual construction rules, not aesthetic adjectives) ==========
# Line weight, shading layers, palette, proportion, eyes, shadow, texture, cel/print.
STYLE_PROMPTS: dict[str, str] = {
    "dragonball": (
        "thick 3-4px black outline, 2-tone cel shading, single hard shadow block, "
        "flat color fill no gradient, bold simplified anatomy, large round eyes with single highlight, "
        "spiky simplified hair shape, manga panel look, limited color palette, no texture"
    ),
    "slamdunk": (
        "medium 2-3px outline, 2-layer cel shading, angular shadow blocks, "
        "flat color regions, athletic proportion, sharp determined eyes with 2-tone, "
        "dynamic pose construction, sports manga linework, no gradient, no soft shadow"
    ),
    "sailor_moon": (
        "clean 2px outline, 2-tone cel shading, sparkle and star motifs, "
        "flat pastel color fill, large eyes with multiple highlights, magical girl proportion, "
        "ribbon and simple costume shapes, no gradient, decorative but flat, 90s anime cel"
    ),
    "pokemon": (
        "soft 1-2px outline, 2-layer cel shading, rounded shadow shapes, "
        "flat saturated color, cute proportion, large simple eyes with one highlight, "
        "rounded silhouette, no fur texture, creature design flat color, no gradient"
    ),
    "dooly": (
        "thick 2-3px outline, 1-tone flat fill, minimal or no shadow, "
        "flat color only, simple rounded anatomy, dot eyes or very simple eyes, "
        "comic strip style, limited palette, no gradient, no shading layers"
    ),
    "mazinger": (
        "thick mechanical outline 3-4px, 2-tone cel shading on metal, hard angular shadows, "
        "flat color with edge highlight only, robot proportion, simplified mecha silhouette, "
        "retro super robot style, no gradient, blocky shadow geometry"
    ),
    "shinchan": (
        "rough 2px outline, 1-2 tone flat shading, simple angular shadow, "
        "flat color fill, exaggerated cartoon proportion, minimal eye detail, "
        "gag manga style, limited palette, no gradient, simple shape-based"
    ),
    "pixel_art": (
        "single pixel line width, 0 gradient, 1-2 flat color layers per region, "
        "hard pixel edges only, no anti-aliasing, max 16 color palette, "
        "sprite-like composition, square pixels, no soft shadow, no subsurface, "
        "low resolution aesthetic, tile-aligned shapes"
    ),
    # Aliases
    "pixel art": (
        "single pixel line width, 0 gradient, 1-2 flat color layers per region, "
        "hard pixel edges only, no anti-aliasing, max 16 color palette, "
        "sprite-like composition, square pixels, no soft shadow, no subsurface, "
        "low resolution aesthetic, tile-aligned shapes"
    ),
}

# Style-specific negative additions (on top of BASE_NEGATIVE)
NEGATIVE_BY_STYLE: dict[str, str] = {
    "dragonball": "soft shading, gradient, realistic muscle, photorealistic",
    "slamdunk": "soft shading, gradient, 3D, photorealistic",
    "sailor_moon": "dark gritty, gradient, photorealistic",
    "pokemon": "realistic fur, gradient, photorealistic",
    "dooly": "complex shading, gradient, detailed texture",
    "mazinger": "organic texture, fur, gradient, photorealistic",
    "shinchan": "realistic, gradient, detailed anatomy",
    "pixel_art": "gradient, anti-aliasing, smooth edges, high resolution, blur, soft shadow, 3D, voxel",
    "pixel art": "gradient, anti-aliasing, smooth edges, high resolution, blur, soft shadow, 3D, voxel",
}

# ========== GENERATION RULES (resolution, steps, cfg) ==========
# pixel_art: 128, steps 32, cfg 7; else: 768, steps 28-32, cfg 6.5-7
GENERATION_RULES: dict[str, dict[str, Any]] = {
    "pixel_art": {"max_side": 128, "steps": 32, "guidance_scale": 7.0},
    "pixel art": {"max_side": 128, "steps": 32, "guidance_scale": 7.0},
    "dragonball": {"max_side": 768, "steps": 30, "guidance_scale": 6.5},
    "slamdunk": {"max_side": 768, "steps": 30, "guidance_scale": 6.5},
    "sailor_moon": {"max_side": 768, "steps": 30, "guidance_scale": 6.5},
    "pokemon": {"max_side": 768, "steps": 30, "guidance_scale": 6.5},
    "dooly": {"max_side": 768, "steps": 28, "guidance_scale": 6.5},
    "mazinger": {"max_side": 768, "steps": 30, "guidance_scale": 7.0},
    "shinchan": {"max_side": 768, "steps": 28, "guidance_scale": 6.5},
}

# Backward compat for code that imports STYLE_TEMPLATES
STYLE_TEMPLATES = STYLE_PROMPTS

ALLOWED_STYLE_KEYS = [
    "dragonball",
    "slamdunk",
    "sailor_moon",
    "pokemon",
    "dooly",
    "mazinger",
    "shinchan",
    "pixel_art",
    "pixel art",
]


def _normalize_style(style: str | None) -> str | None:
    if style is None or not str(style).strip():
        return None
    key = str(style).strip().lower().replace(" ", "_")
    if key in STYLE_PROMPTS:
        return key
    if key.replace("_", " ") in STYLE_PROMPTS:
        return key.replace("_", " ")
    for k in STYLE_PROMPTS:
        if k.replace(" ", "_") == key:
            return k
    return style.strip().lower().replace(" ", "_")


def get_prompt_config() -> dict[str, Any]:
    """Return full structured config for API/docs."""
    return {
        "base_prompt": BASE_PROMPT,
        "base_negative": BASE_NEGATIVE,
        "styles": dict(STYLE_PROMPTS),
        "generation_rules": dict(GENERATION_RULES),
    }


def get_generation_rules(style_key: str | None) -> dict[str, Any]:
    """Get max_side, steps, guidance_scale for a style. Defaults: 768, 30, 6.5."""
    default = {"max_side": 768, "steps": 30, "guidance_scale": 6.5}
    if not style_key:
        return default
    key = _normalize_style(style_key)
    if not key:
        return default
    return GENERATION_RULES.get(key, default)


# 사용자가 편집 지시를 비워 둘 때 사용할 기본 서술 (2D 캐릭터 재해석용)
DEFAULT_USER_PROMPT = "a small white dog character"

def build_prompt(user_prompt: str, style: str | None = None, raw_prompt: bool = False) -> str:
    """
    Final prompt = user_prompt + BASE_PROMPT + style construction rules.
    Never describes a real dog; always redesigned animated character.
    사용자가 비우면 DEFAULT_USER_PROMPT 사용 (400 방지).
    """
    text = (user_prompt or "").strip()
    if not text:
        text = DEFAULT_USER_PROMPT
    if raw_prompt:
        return text
    style_key = _normalize_style(style)
    if style_key is None:
        return f"{text}, {BASE_PROMPT}"
    template = STYLE_PROMPTS.get(style_key)
    if not template:
        return f"{text}, {BASE_PROMPT}"
    return f"{text}, {BASE_PROMPT}, {template}"


def build_negative_prompt(style: str | None, raw_prompt: bool = False) -> str:
    """Final negative = BASE_NEGATIVE + style-specific negative."""
    if raw_prompt:
        return ""
    style_key = _normalize_style(style)
    style_neg = NEGATIVE_BY_STYLE.get(style_key, "") if style_key else ""
    if style_neg:
        return f"{BASE_NEGATIVE}, {style_neg}"
    return BASE_NEGATIVE


def get_allowed_style_keys() -> list[str]:
    """API validation: list of allowed style keys (no duplicates for display)."""
    seen: set[str] = set()
    out: list[str] = []
    for k in ALLOWED_STYLE_KEYS:
        if k not in seen:
            seen.add(k)
            out.append(k)
    return out
