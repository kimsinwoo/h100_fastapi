"""
STYLE_REGISTRY: style → model_key, prompts, defaults, LoRA.
User-provided cfg/steps override; if not provided use style defaults.
"""
from __future__ import annotations

from typing import TypedDict

from app.sdxl_prod.model_keys import (
    MODEL_KEY_ANIMAGINE_XL,
    MODEL_KEY_DREAMSHAPER_XL,
    MODEL_KEY_JUGGERNAUT_XL,
    MODEL_KEY_PIXELART_XL,
    MODEL_KEY_REALVIS_XL,
    MODEL_KEY_SDXL_BASE,
)

SAMPLER_EULER_A = "euler_a"
SAMPLER_DPMPP_2M_KARRAS = "dpmpp_2m_karras"


class StrengthRangeT(TypedDict):
    min: float
    max: float


class StyleConfigT(TypedDict):
    model_key: str
    positive_prompt_template: str
    negative_prompt_template: str
    default_cfg: float
    default_steps: int
    default_sampler: str
    recommended_strength_range: StrengthRangeT
    optional_lora_key: str  # empty string = no LoRA


def _sr(min_val: float, max_val: float) -> StrengthRangeT:
    return StrengthRangeT(min=min_val, max=max_val)


STYLE_REGISTRY: dict[str, StyleConfigT] = {
    "anime": StyleConfigT(
        model_key=MODEL_KEY_ANIMAGINE_XL,
        positive_prompt_template=(
            "masterpiece, best quality, 2D anime, clean lineart, cel shading, "
            "{subject}, "
            "vibrant, sharp, studio quality"
        ),
        negative_prompt_template=(
            "lowres, bad anatomy, bad hands, text, error, blurry, 3d, photo, realistic"
        ),
        default_cfg=7.0,
        default_steps=28,
        default_sampler=SAMPLER_DPMPP_2M_KARRAS,
        recommended_strength_range=_sr(0.45, 0.58),
        optional_lora_key="",
    ),
    "realistic": StyleConfigT(
        model_key=MODEL_KEY_REALVIS_XL,
        positive_prompt_template=(
            "RAW photo, DSLR, 8k, natural skin texture, cinematic lighting, "
            "{subject}, "
            "sharp focus, high resolution, professional"
        ),
        negative_prompt_template=(
            "cartoon, anime, illustration, painting, drawing, sketch, "
            "plastic, oversharpened, low quality"
        ),
        default_cfg=7.0,
        default_steps=28,
        default_sampler=SAMPLER_DPMPP_2M_KARRAS,
        recommended_strength_range=_sr(0.45, 0.58),
        optional_lora_key="",
    ),
    "cinematic": StyleConfigT(
        model_key=MODEL_KEY_JUGGERNAUT_XL,
        positive_prompt_template=(
            "cinematic still, film grain, dramatic lighting, anamorphic, "
            "{subject}, "
            "8k, professional color grading"
        ),
        negative_prompt_template=(
            "cartoon, anime, flat lighting, amateur, overexposed"
        ),
        default_cfg=7.0,
        default_steps=28,
        default_sampler=SAMPLER_DPMPP_2M_KARRAS,
        recommended_strength_range=_sr(0.45, 0.56),
        optional_lora_key="",
    ),
    "3d_render": StyleConfigT(
        model_key=MODEL_KEY_REALVIS_XL,
        positive_prompt_template=(
            "3D render, Unreal Engine, PBR, ray tracing, "
            "{subject}, "
            "high poly, detailed, 8k"
        ),
        negative_prompt_template=(
            "2D, flat, sketch, painting, photograph"
        ),
        default_cfg=7.0,
        default_steps=28,
        default_sampler=SAMPLER_DPMPP_2M_KARRAS,
        recommended_strength_range=_sr(0.48, 0.58),
        optional_lora_key="",
    ),
    "cyberpunk": StyleConfigT(
        model_key=MODEL_KEY_DREAMSHAPER_XL,
        positive_prompt_template=(
            "cyberpunk, neon noir, futuristic, neon lights, rain, "
            "{subject}, "
            "blade runner style, high contrast"
        ),
        negative_prompt_template=(
            "daylight, nature, pastoral, vintage, watercolor"
        ),
        default_cfg=7.0,
        default_steps=28,
        default_sampler=SAMPLER_DPMPP_2M_KARRAS,
        recommended_strength_range=_sr(0.45, 0.58),
        optional_lora_key="",
    ),
    "fantasy_art": StyleConfigT(
        model_key=MODEL_KEY_DREAMSHAPER_XL,
        positive_prompt_template=(
            "fantasy art, epic, detailed armor, magical, ethereal, "
            "{subject}, "
            "concept art, Greg Rutkowski style"
        ),
        negative_prompt_template=(
            "modern, photograph, sci-fi, minimalist"
        ),
        default_cfg=7.0,
        default_steps=28,
        default_sampler=SAMPLER_DPMPP_2M_KARRAS,
        recommended_strength_range=_sr(0.45, 0.58),
        optional_lora_key="",
    ),
    # alias: fantasy → same as fantasy_art (dreamshaper_xl)
    "fantasy": StyleConfigT(
        model_key=MODEL_KEY_DREAMSHAPER_XL,
        positive_prompt_template=(
            "fantasy art, epic, detailed armor, magical, ethereal, "
            "{subject}, "
            "concept art, Greg Rutkowski style"
        ),
        negative_prompt_template=(
            "modern, photograph, sci-fi, minimalist"
        ),
        default_cfg=7.0,
        default_steps=28,
        default_sampler=SAMPLER_DPMPP_2M_KARRAS,
        recommended_strength_range=_sr(0.45, 0.58),
        optional_lora_key="",
    ),
    "watercolor": StyleConfigT(
        model_key=MODEL_KEY_SDXL_BASE,
        positive_prompt_template=(
            "watercolor painting, paper texture, pigment bleeding, "
            "{subject}, "
            "soft edges, traditional medium"
        ),
        negative_prompt_template=(
            "3d render, digital, vector, oil painting, photograph"
        ),
        default_cfg=7.0,
        default_steps=28,
        default_sampler=SAMPLER_DPMPP_2M_KARRAS,
        recommended_strength_range=_sr(0.45, 0.58),
        optional_lora_key="watercolor",
    ),
    "oil_painting": StyleConfigT(
        model_key=MODEL_KEY_SDXL_BASE,
        positive_prompt_template=(
            "oil painting on canvas, impasto, visible brushstrokes, "
            "{subject}, "
            "classical, museum quality"
        ),
        negative_prompt_template=(
            "digital, photograph, anime, pixel art, sketch"
        ),
        default_cfg=7.0,
        default_steps=28,
        default_sampler=SAMPLER_DPMPP_2M_KARRAS,
        recommended_strength_range=_sr(0.45, 0.58),
        optional_lora_key="oil_painting",
    ),
    "sketch": StyleConfigT(
        model_key=MODEL_KEY_SDXL_BASE,
        positive_prompt_template=(
            "pencil sketch, graphite, cross-hatching, paper texture, "
            "{subject}, "
            "monochrome, fine art sketch"
        ),
        negative_prompt_template=(
            "color, painted, 3d, photograph, digital paint"
        ),
        default_cfg=7.0,
        default_steps=28,
        default_sampler=SAMPLER_DPMPP_2M_KARRAS,
        recommended_strength_range=_sr(0.45, 0.58),
        optional_lora_key="sketch",
    ),
    "pixel_art": StyleConfigT(
        model_key=MODEL_KEY_PIXELART_XL,
        positive_prompt_template=(
            "pixel art, 2D sprite, flat, 16-bit style, square pixels, "
            "{subject}, "
            "no anti-aliasing, limited palette"
        ),
        negative_prompt_template=(
            "voxel, 3D, Minecraft, smooth, gradient, realistic, photograph"
        ),
        default_cfg=5.0,
        default_steps=20,
        default_sampler=SAMPLER_EULER_A,
        recommended_strength_range=_sr(0.25, 0.40),
        optional_lora_key="",
    ),
}

DEFAULT_STYLE_KEY = "realistic"


def get_style_config(style_key: str) -> StyleConfigT:
    k = style_key.lower().strip().replace(" ", "_")
    return STYLE_REGISTRY.get(k, STYLE_REGISTRY[DEFAULT_STYLE_KEY])


def list_style_keys() -> list[str]:
    return sorted(STYLE_REGISTRY.keys())
