"""
Expert Image Prompt Engineer — style-specific Positive/Negative for Z-Image-Turbo.
Synced with .cursor/rules/image-prompt-engineer.mdc
"""

from __future__ import annotations

# Style-specific Positive (Final Prompt = User Content + these)
STYLE_PROMPTS: dict[str, str] = {
    "anime": (
        "high-quality 2D anime, flat cel shading, clean line art, vibrant colors, "
        "expressive eyes, hand-drawn look, Makoto Shinkai or Studio Ghibli aesthetic, "
        "4k resolution, crisp edges"
    ),
    "realistic": (
        "photorealistic, 8k UHD, shot on 35mm lens, f/1.8, RAW photo, Fujifilm, "
        "highly detailed skin texture, pores, fine hair, natural lighting, "
        "sharp focus, masterpiece"
    ),
    "watercolor": (
        "authentic watercolor painting, wet-on-wet technique, visible paper texture, "
        "soft bleeding edges, paint drips, hand-painted, delicate layers, artistic strokes"
    ),
    "cyberpunk": (
        "high-contrast cyberpunk aesthetic, neon blue and magenta lighting, "
        "futuristic atmosphere, rainy streets, volumetric fog, cinematic lighting, "
        "sharp details, high-tech vibe"
    ),
    "oil painting": (
        "impasto oil painting, thick brushstrokes, visible canvas texture, "
        "heavy paint layers, Chiaroscuro lighting, classical art style, "
        "rich pigments, museum quality"
    ),
    "sketch": (
        "graphite pencil sketch, charcoal drawing, hand-drawn on textured paper, "
        "cross-hatching, rough artistic lines, high contrast, minimalist, "
        "HB/2B pencil texture"
    ),
    "cinematic": (
        "cinematic film still, anamorphic lens, 2.35:1 aspect ratio, "
        "professional color grading, depth of field, dramatic lighting, "
        "epic composition, film grain, Hollywood look"
    ),
    "fantasy art": (
        "epic fantasy illustration, intricate details, magical atmosphere, "
        "glowing particles, digital painting, Greg Rutkowski style, "
        "heroic composition, high-res"
    ),
    "pixel art": (
        "Strictly 2D, flat, 16-bit, SNES style, clean pixels, aliased, "
        "sprite, non-isometric, no depth, side-scroller character, "
        "limited palette, bold black outlines"
    ),
    "3d render": (
        "Octane render, Ray tracing, Unreal Engine 5, 8k, volumetric lighting, "
        "subsurface scattering, PBR materials, high-poly model, "
        "smooth surfaces, hyper-detailed"
    ),
}

# Style-specific Negative
STYLE_NEGATIVE_PROMPTS: dict[str, str] = {
    "anime": (
        "3D, render, CGI, realistic, photographic, blurry, grainy, "
        "messy lines, 3D model look, gradient hair (unless specified), low resolution"
    ),
    "realistic": (
        "drawing, painting, cartoon, anime, 3d render, CGI, plastic skin, "
        "airbrushed, unnatural eyes, distorted limbs, over-saturated, smooth texture"
    ),
    "watercolor": (
        "sharp digital lines, solid fills, 3d, realistic photo, oil painting, "
        "plastic texture, clean edges, neon colors, digital gradient"
    ),
    "cyberpunk": (
        "bright daylight, natural scenery, rustic, vintage, pastel colors, "
        "soft lighting, low contrast, simplistic"
    ),
    "oil painting": (
        "flat, smooth, digital art, anime, vector, 2d illustration, "
        "clean lines, thin paint, photographic"
    ),
    "sketch": (
        "color, digital paint, 3d, render, smooth gradients, "
        "clean vector lines, photographic, blurry"
    ),
    "cinematic": (
        "cartoon, anime, flat lighting, amateur photo, phone camera, "
        "bright and cheerful, low contrast, 2d"
    ),
    "fantasy art": (
        "modern, realistic photo, sci-fi, pixel art, low detail, "
        "simple background, 3d voxel"
    ),
    "pixel art": (
        "Minecraft, voxel, 3D blocks, lego, isometric 3D, depth, volume, "
        "smooth, anti-aliased, gradients, soft shading, realistic, render"
    ),
    "3d render": (
        "2d, flat, sketch, painting, pixel, low-poly, grainy, "
        "cartoon lines, hand-drawn"
    ),
}

BASE_NEGATIVE = "blurry, low quality, distorted, watermark, text"

DEFAULT_STYLE = "realistic"

STYLE_PRESETS: dict[str, str] = {
    "anime": "Anime (2D Cel-Shaded, Makoto Shinkai / Ghibli)",
    "realistic": "Realistic (Photographic, 8k UHD)",
    "watercolor": "Watercolor (Traditional)",
    "cyberpunk": "Cyberpunk (Futuristic)",
    "oil painting": "Oil Painting (Classical)",
    "sketch": "Sketch (Hand-drawn)",
    "cinematic": "Cinematic (Movie Scene)",
    "fantasy art": "Fantasy Art (Digital Illustration)",
    "pixel art": "Pixel Art (2D flat, SNES, non-isometric)",
    "3d render": "3D Render (High-End CGI)",
}


def get_style_prompt(style_key: str) -> str:
    key = style_key.strip().lower()
    return STYLE_PROMPTS.get(key, STYLE_PROMPTS[DEFAULT_STYLE])


def get_style_negative_prompt(style_key: str) -> str:
    key = style_key.strip().lower()
    neg = STYLE_NEGATIVE_PROMPTS.get(key, STYLE_NEGATIVE_PROMPTS[DEFAULT_STYLE])
    return f"{BASE_NEGATIVE}, {neg}"


def merge_prompt(style_key: str, custom_prompt: str | None) -> str:
    """Final Prompt = (User Content first), (Style Positive)."""
    style_text = get_style_prompt(style_key)
    if not (custom_prompt and custom_prompt.strip()):
        return style_text
    return f"{custom_prompt.strip()}, {style_text}"
