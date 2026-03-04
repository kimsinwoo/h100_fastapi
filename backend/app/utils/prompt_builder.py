"""
Stylized 2D character redesign for ANY pet species.
Subject: small pet animal (dog, cat, rabbit, bird, etc.) → simplified 2D animated character.
Never preserve biological realism. No fur/feather/skin realism, no 3D lighting.
"""

from __future__ import annotations

from typing import Any

# ========== BASE PROMPT (species-agnostic) ==========
# Stylized 2D animated character redesign; subject = "small pet animal character"
BASE_PROMPT = (
    "stylized 2D animated character redesign of a small pet animal character, "
    "simplified shape language, clear silhouette, flat color areas, limited shading layers, "
    "no biological realism, no realistic muscle definition, no real texture replication, "
    "style-locked rendering, not a photograph, animated character form only"
)

# ========== BASE NEGATIVE ==========
BASE_NEGATIVE = (
    "photorealistic, real fur, real feathers, real skin texture, "
    "cinematic lighting, 3D rendering, ray tracing, depth of field, subsurface scattering, "
    "hyper detailed, realistic anatomy, real muscle structure, high frequency texture, HDR"
)

# ========== SPECIES SUBJECT (주어 명시: 고양이/강아지 구분 확실히) ==========
# 프롬프트 맨 앞에 올려 모델이 종을 명확히 인식하도록 함.
SPECIES_SUBJECT: dict[str, str] = {
    "dog": "dog character",
    "cat": "cat character",
    "rabbit": "rabbit character",
    "hamster": "hamster character",
    "ferret": "ferret character",
    "bird": "bird character",
    "turtle": "turtle character",
    "reptile": "reptile character",
    "pet": "small pet animal character",
}

# ========== SPECIES ADAPTATION LAYER ==========
# Silhouette, ear, eye, limb, tail, texture simplification per species.
SPECIES_MODIFIERS: dict[str, str] = {
    "dog": (
        "rounded muzzle silhouette, simplified ear blocks, tail as single curved shape, "
        "no fur strands, simplified limb blocks, no wet nose detail"
    ),
    "cat": (
        "triangular ear blocks, sharper eye shape, tail as smooth tapered curve, "
        "whiskers as simple lines only no realism, simplified limb, no fur strands"
    ),
    "rabbit": (
        "elongated ear shapes, minimal facial detail, round body mass, "
        "simplified feet, no fur texture, clear silhouette"
    ),
    "hamster": (
        "round compact silhouette, small simplified ears, minimal limb detail, "
        "no fur strands, simple cheek shape, no realistic fur texture"
    ),
    "ferret": (
        "elongated body as simplified shape, rounded ears, tapered tail as single curve, "
        "no fur detail, simplified limbs, clear silhouette"
    ),
    "bird": (
        "beak as flat geometric wedge, wings as layered flat shapes, no feather detail, "
        "feet simplified to minimal shapes, round or simplified head, no plumage texture"
    ),
    "turtle": (
        "shell as simple dome or rounded shape, head and limbs as simplified blocks, "
        "no scale texture, no realistic skin, clear silhouette"
    ),
    "reptile": (
        "simplified body silhouette, head as geometric shape, no scale texture, "
        "limbs as simple shapes, no realistic skin detail"
    ),
    "pet": (
        "simplified animal silhouette, clear shape language, no fur or feather strands, "
        "minimal limb and tail detail, flat color regions, readable at small size"
    ),
}

# ========== SPECIES NEGATIVE (반대 종 회피: 고양이일 때 강아지 형태 차단, 강아지일 때 고양이 형태 차단) ==========
SPECIES_NEGATIVE_AVOID: dict[str, str] = {
    "dog": "cat, feline, cat-like, triangular cat ears, slit pupils, cat silhouette",
    "cat": "dog, canine, dog-like, rounded dog muzzle, floppy dog ears, dog silhouette",
    "rabbit": "dog, cat, canine, feline, long dog muzzle",
    "hamster": "dog, cat, long ears, long muzzle",
    "ferret": "dog, cat, bird, beak",
    "bird": "dog, cat, mammal, fur, four legs",
    "turtle": "dog, cat, fur, ears",
    "reptile": "dog, cat, fur, fluffy",
    "pet": "",
}

# ========== STYLE LAYER (construction rules) ==========
# Outline thickness, shading layer count, color palette, eye rendering, highlight, shadow geometry.
STYLE_PROMPTS: dict[str, str] = {
    "dragonball": (
        "thick 3-4px black outline, 2-tone cel shading, single hard shadow block, "
        "flat color fill no gradient, bold simplified anatomy, large round eyes with single highlight, "
        "limited color palette, no texture, manga panel look"
    ),
    "slamdunk": (
        "medium 2-3px outline, 2-layer cel shading, angular shadow blocks, "
        "flat color regions, sharp determined eyes with 2-tone, "
        "dynamic pose construction, sports manga linework, no gradient, no soft shadow"
    ),
    "sailor_moon": (
        "clean 2px outline, 2-tone cel shading, sparkle and star motifs, "
        "flat pastel color fill, large eyes with multiple highlights, "
        "ribbon and simple costume shapes, no gradient, 90s anime cel"
    ),
    "pokemon": (
        "soft 1-2px outline, 2-layer cel shading, rounded shadow shapes, "
        "flat saturated color, cute proportion, large simple eyes with one highlight, "
        "rounded silhouette, creature design flat color, no gradient"
    ),
    "dooly": (
        "thick 2-3px outline, 1-tone flat fill, minimal or no shadow, "
        "flat color only, simple rounded anatomy, dot eyes or very simple eyes, "
        "comic strip style, limited palette, no gradient, no shading layers"
    ),
    "mazinger": (
        "thick mechanical outline 3-4px, 2-tone cel shading, hard angular shadows, "
        "flat color with edge highlight only, simplified mecha silhouette, "
        "retro super robot style, no gradient, blocky shadow geometry"
    ),
    "shinchan": (
        "rough 2px outline, 1-2 tone flat shading, simple angular shadow, "
        "flat color fill, exaggerated cartoon proportion, minimal eye detail, "
        "gag manga style, limited palette, no gradient, simple shape-based"
    ),
    "pixel_art": (
        "single pixel line width, 0 gradient, pixel cluster shading only, "
        "hard pixel edges only, no anti-aliasing, max 16 colors, no soft shading, "
        "sprite composition, silhouette reads clearly at 1x zoom, square pixels"
    ),
    "pixel art": (
        "single pixel line width, 0 gradient, pixel cluster shading only, "
        "hard pixel edges only, no anti-aliasing, max 16 colors, no soft shading, "
        "sprite composition, silhouette reads clearly at 1x zoom, square pixels"
    ),
}

NEGATIVE_BY_STYLE: dict[str, str] = {
    "dragonball": "soft shading, gradient, realistic muscle, photorealistic",
    "slamdunk": "soft shading, gradient, 3D, photorealistic",
    "sailor_moon": "dark gritty, gradient, photorealistic",
    "pokemon": "realistic fur, realistic feathers, gradient, photorealistic",
    "dooly": "complex shading, gradient, detailed texture",
    "mazinger": "organic texture, fur, gradient, photorealistic",
    "shinchan": "realistic, gradient, detailed anatomy",
    "pixel_art": (
        "gradient, anti-aliasing, smooth edges, high resolution, blur, soft shadow, 3D, voxel, "
        "edge artifact, chromatic aberration, muddy, desaturated strip"
    ),
    "pixel art": (
        "gradient, anti-aliasing, smooth edges, high resolution, blur, soft shadow, 3D, voxel, "
        "edge artifact, chromatic aberration, muddy, desaturated strip"
    ),
}

# ========== GENERATION RULES ==========
# pixel_art: 128x128, steps 32, cfg 7, nearest upscale only. Else: 768, steps 28-32, cfg 6.5-7
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

ALLOWED_SPECIES_KEYS = list(SPECIES_MODIFIERS.keys())


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


def _normalize_species(species: str | None) -> str | None:
    if species is None or not str(species).strip():
        return None
    key = str(species).strip().lower().replace(" ", "_")
    return key if key in SPECIES_MODIFIERS else ("pet" if key else None)


def get_prompt_config() -> dict[str, Any]:
    """Full structured config: base_prompt, base_negative, species_modifiers, styles, generation_rules."""
    return {
        "base_prompt": BASE_PROMPT,
        "base_negative": BASE_NEGATIVE,
        "species_modifiers": dict(SPECIES_MODIFIERS),
        "styles": dict(STYLE_PROMPTS),
        "generation_rules": dict(GENERATION_RULES),
    }


def get_generation_rules(style_key: str | None) -> dict[str, Any]:
    """Get max_side, steps, guidance_scale for a style. pixel_art: 128, 32, 7; else 768, 28-30, 6.5-7."""
    default = {"max_side": 768, "steps": 30, "guidance_scale": 6.5}
    if not style_key:
        return default
    key = _normalize_style(style_key)
    if not key:
        return default
    return GENERATION_RULES.get(key, default)


# Default subject when user leaves prompt empty (generic small pet animal)
DEFAULT_USER_PROMPT = "small pet animal character"


def build_prompt(
    user_prompt: str,
    style: str | None = None,
    species: str | None = None,
    raw_prompt: bool = False,
) -> str:
    """
    Final prompt = (종 주어) + user_prompt + BASE_PROMPT + species_modifier + style rules.
    species가 있으면 주어를 "cat character" / "dog character" 등으로 명시해 고양이/강아지 구분을 확실히 함.
    """
    text = (user_prompt or "").strip()
    species_key = _normalize_species(species)
    if not raw_prompt and species_key and species_key in SPECIES_SUBJECT:
        subject = SPECIES_SUBJECT[species_key]
        if not text:
            text = subject
        else:
            text = f"{subject}, {text}"
    elif not text:
        text = DEFAULT_USER_PROMPT
    if raw_prompt:
        return text
    parts = [text, BASE_PROMPT]
    if species_key and species_key in SPECIES_MODIFIERS:
        parts.append(SPECIES_MODIFIERS[species_key])
    style_key = _normalize_style(style)
    if style_key and style_key in STYLE_PROMPTS:
        parts.append(STYLE_PROMPTS[style_key])
    return ", ".join(parts)


def build_negative_prompt(
    style: str | None,
    species: str | None = None,
    raw_prompt: bool = False,
) -> str:
    """Final negative = BASE_NEGATIVE + species cross-avoid + style-specific negative."""
    if raw_prompt:
        return ""
    parts = [BASE_NEGATIVE]
    species_key = _normalize_species(species)
    if species_key and species_key in SPECIES_NEGATIVE_AVOID and SPECIES_NEGATIVE_AVOID[species_key]:
        parts.append(SPECIES_NEGATIVE_AVOID[species_key])
    style_key = _normalize_style(style)
    style_neg = NEGATIVE_BY_STYLE.get(style_key, "") if style_key else ""
    if style_neg:
        parts.append(style_neg)
    return ", ".join(p for p in parts if p)


def get_allowed_style_keys() -> list[str]:
    """API: allowed style keys (no duplicate display)."""
    seen: set[str] = set()
    out: list[str] = []
    for k in ALLOWED_STYLE_KEYS:
        if k not in seen:
            seen.add(k)
            out.append(k)
    return out


def get_allowed_species_keys() -> list[str]:
    """API: allowed species keys for species_modifiers."""
    return list(ALLOWED_SPECIES_KEYS)
