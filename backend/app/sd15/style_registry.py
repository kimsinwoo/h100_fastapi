"""
Structured style configuration for SD 1.5 img2img.
Each style has distinct prompts, defaults, and recommended ranges to avoid style blending.
"""
from __future__ import annotations

from typing import TypedDict

# Sampler names used with diffusers (EulerAncestralDiscreteScheduler, DPMSolverMultistepScheduler)
SAMPLER_EULER_A = "euler_a"
SAMPLER_DPMPP_2M_KARRAS = "dpmpp_2m_karras"


class StrengthRange(TypedDict):
    min: float
    max: float


class StyleConfig(TypedDict):
    positive_prompt_template: str
    negative_prompt_template: str
    default_cfg: float
    default_steps: int
    default_sampler: str
    optional_lora: str  # style key used to resolve LoRA path, e.g. "pixel_art"
    recommended_strength_range: StrengthRange
    enable_upscale: bool  # Pixel Art disables
    output_grayscale: bool  # Sketch can force grayscale


def _sr(min_val: float, max_val: float) -> StrengthRange:
    return StrengthRange(min=min_val, max=max_val)


# ---------------------------------------------------------------------------
# Style definitions: visually distinct, strong stylistic identity
# ---------------------------------------------------------------------------

STYLE_REGISTRY: dict[str, StyleConfig] = {
    "anime": StyleConfig(
        positive_prompt_template=(
            "STRICT 2D Japanese anime frame, "
            "clean lineart, cel shading, flat color shading, "
            "anime lighting, vibrant but controlled colors, "
            "{subject}, "
            "high detail lineart, crisp edges, studio quality anime frame, sharp outlines"
        ),
        negative_prompt_template=(
            "photo, realistic, 3d render, skin texture, pore detail, "
            "film grain, cinematic photography, lens blur, "
            "western cartoon, illustration, painting"
        ),
        default_cfg=7.5,
        default_steps=28,
        default_sampler=SAMPLER_DPMPP_2M_KARRAS,
        optional_lora="anime",
        recommended_strength_range=_sr(0.48, 0.58),
        enable_upscale=True,
        output_grayscale=False,
    ),
    "realistic": StyleConfig(
        positive_prompt_template=(
            "RAW professional photograph, DSLR, "
            "natural skin texture, true-to-life color grading, "
            "cinematic lighting, high dynamic range, "
            "subtle film grain, natural imperfections, realistic lens behavior, "
            "shot on Phase One, 80mm f/2.8, medium format, "
            "{subject}, "
            "ultra detailed, sharp focus, high resolution, fine texture detail, real skin pores, balanced exposure"
        ),
        negative_prompt_template=(
            "cartoon, anime, illustration, painting, cgi, 3d render, "
            "plastic skin, smooth skin, oversharpened, unrealistic lighting, "
            "drawing, sketch, watercolor, oil painting, low quality, blurry, overprocessed, exaggerated features"
        ),
        default_cfg=8.0,
        default_steps=30,
        default_sampler=SAMPLER_DPMPP_2M_KARRAS,
        optional_lora="realistic",
        recommended_strength_range=_sr(0.46, 0.56),
        enable_upscale=True,
        output_grayscale=False,
    ),
    "watercolor": StyleConfig(
        positive_prompt_template=(
            "Traditional watercolor on cotton paper, "
            "hand-painted watercolor, paper texture visible, "
            "wet-on-wet technique, natural pigment bleeding, transparent washes, "
            "soft organic edges, no digital artifacts, "
            "{subject}, "
            "Arches cold press, fine art quality"
        ),
        negative_prompt_template=(
            "3d render, photorealistic, photograph, digital art, vector, "
            "oil painting, acrylic, neon, CGI, smooth gradient, "
            "plastic, glossy, sharp digital edges"
        ),
        default_cfg=7.5,
        default_steps=30,
        default_sampler=SAMPLER_DPMPP_2M_KARRAS,
        optional_lora="watercolor",
        recommended_strength_range=_sr(0.48, 0.58),
        enable_upscale=True,
        output_grayscale=False,
    ),
    "cyberpunk": StyleConfig(
        positive_prompt_template=(
            "Cyberpunk neon noir, futuristic scene, "
            "blue and magenta neon lighting, wet reflective streets, "
            "volumetric fog, holographic UI glow, high contrast night atmosphere, "
            "{subject}, "
            "cinematic, dramatic, blade runner style"
        ),
        negative_prompt_template=(
            "daylight, nature, watercolor, oil painting, anime, cartoon, "
            "pastel, soft lighting, rustic, vintage, peaceful"
        ),
        default_cfg=7.5,
        default_steps=30,
        default_sampler=SAMPLER_DPMPP_2M_KARRAS,
        optional_lora="cyberpunk",
        recommended_strength_range=_sr(0.48, 0.58),
        enable_upscale=True,
        output_grayscale=False,
    ),
    "oil_painting": StyleConfig(
        positive_prompt_template=(
            "Oil paint on canvas, museum quality oil painting, "
            "impasto brushstrokes, visible canvas weave, "
            "rich layered pigment, chiaroscuro lighting, classical master technique, "
            "{subject}, "
            "fine art, gallery quality"
        ),
        negative_prompt_template=(
            "flat digital, anime, pixel art, sketch, photograph, "
            "watercolor, vector, cel shading, 3d render"
        ),
        default_cfg=7.5,
        default_steps=32,
        default_sampler=SAMPLER_DPMPP_2M_KARRAS,
        optional_lora="oil_painting",
        recommended_strength_range=_sr(0.48, 0.58),
        enable_upscale=True,
        output_grayscale=False,
    ),
    "sketch": StyleConfig(
        positive_prompt_template=(
            "Graphite pencil drawing, high contrast charcoal sketch, "
            "cross-hatching, controlled shading, paper grain texture visible, "
            "monochrome, no color, no paint, no digital effects, "
            "{subject}, "
            "fine art sketch, detailed linework"
        ),
        negative_prompt_template=(
            "color, painted, digital paint, 3d render, photograph, "
            "watercolor, oil, neon, saturated, cel shading"
        ),
        default_cfg=7.5,
        default_steps=28,
        default_sampler=SAMPLER_DPMPP_2M_KARRAS,
        optional_lora="sketch",
        recommended_strength_range=_sr(0.48, 0.58),
        enable_upscale=True,
        output_grayscale=True,
    ),
    "cinematic": StyleConfig(
        positive_prompt_template=(
            "Cinema frame, feature film still, Hollywood grade cinematography, "
            "shot on ARRI Alexa 65, anamorphic 2.39:1, "
            "motivated lighting design, rim light separation, controlled shadows, "
            "volumetric atmosphere, Kodak 2383 film emulation, subtle film grain, "
            "{subject}, "
            "award-winning cinematography quality"
        ),
        negative_prompt_template=(
            "cartoon, anime, flat lighting, amateur, phone camera, pixel art, "
            "watercolor, oil painting, sketch, overexposed, underexposed"
        ),
        default_cfg=7.5,
        default_steps=30,
        default_sampler=SAMPLER_DPMPP_2M_KARRAS,
        optional_lora="cinematic",
        recommended_strength_range=_sr(0.46, 0.54),
        enable_upscale=True,
        output_grayscale=False,
    ),
    "fantasy_art": StyleConfig(
        positive_prompt_template=(
            "Epic fantasy digital illustration, "
            "extremely intricate detailed armor fabric jewelry and weapons, "
            "magical glow, floating particles, luminous ethereal atmosphere, "
            "dragons castles knights wizards, Greg Rutkowski style, "
            "{subject}, "
            "heroic dynamic pose, dramatic composition, 8k, concept art quality"
        ),
        negative_prompt_template=(
            "modern clothing, modern setting, photograph, sci-fi, technology, "
            "pixel art, 3d voxel, Minecraft, realistic portrait, minimalist, cartoon"
        ),
        default_cfg=7.5,
        default_steps=32,
        default_sampler=SAMPLER_DPMPP_2M_KARRAS,
        optional_lora="fantasy_art",
        recommended_strength_range=_sr(0.48, 0.58),
        enable_upscale=True,
        output_grayscale=False,
    ),
    "pixel_art": StyleConfig(
        positive_prompt_template=(
            "PURE 2D PIXEL ART SPRITE, strictly 2D only, "
            "completely flat single plane, zero depth zero perspective zero volume, "
            "16-bit SNES Sega Genesis era sprite style, square blocky pixels only, "
            "no anti-aliasing, hard aliased jagged edges, "
            "single character or object sprite, non-isometric, side view or front view only, "
            "maximum 8 to 16 colors limited palette, bold thick black outline, "
            "{subject}, "
            "retro video game character sprite sheet"
        ),
        negative_prompt_template=(
            "Minecraft, voxel, 3D blocks, lego, isometric 3D, depth, volume, perspective, "
            "round, sculpted, curved, smooth shading, anti-aliased, gradient, "
            "soft shadow, photorealistic, 3D render, CGI, ray tracing, realistic, "
            "photograph, smooth edges, blur, depth of field, bokeh, "
            "oil painting, watercolor, anime, cartoon"
        ),
        default_cfg=5.0,
        default_steps=25,
        default_sampler=SAMPLER_EULER_A,
        optional_lora="pixel_art",
        recommended_strength_range=_sr(0.23, 0.38),
        enable_upscale=False,
        output_grayscale=False,
    ),
    "3d_render": StyleConfig(
        positive_prompt_template=(
            "AAA CGI render, Unreal Engine 5 cinematic render, "
            "Nanite geometry, Lumen global illumination, path traced lighting, "
            "PBR physically based materials, high-polygon smooth topology, "
            "volumetric god rays, subsurface scattering on skin, "
            "{subject}, "
            "photorealistic CGI, 8k resolution"
        ),
        negative_prompt_template=(
            "2D, flat illustration, sketch, painting, pixel art, hand-drawn, "
            "low-poly, faceted, grainy, cartoon outline, cel shading, "
            "watercolor, oil painting, photograph, anime"
        ),
        default_cfg=7.5,
        default_steps=30,
        default_sampler=SAMPLER_DPMPP_2M_KARRAS,
        optional_lora="3d_render",
        recommended_strength_range=_sr(0.50, 0.58),
        enable_upscale=True,
        output_grayscale=False,
    ),
}

DEFAULT_STYLE_KEY = "realistic"


def get_style_config(style_key: str) -> StyleConfig:
    key = style_key.lower().strip().replace(" ", "_")
    if key in STYLE_REGISTRY:
        return STYLE_REGISTRY[key]
    return STYLE_REGISTRY[DEFAULT_STYLE_KEY]


def list_style_keys() -> list[str]:
    return sorted(STYLE_REGISTRY.keys())


def is_valid_style(style_key: str) -> bool:
    key = style_key.lower().strip().replace(" ", "_")
    return key in STYLE_REGISTRY
