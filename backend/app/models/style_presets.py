"""
Expert Image Prompt Engineer — style-specific positive/negative prompts for Z-Image-Turbo.
Pixel Art: strictly 2D, flat, non-isometric to avoid Minecraft/voxel.
"""

from __future__ import annotations

# Style-specific positive keywords (auto-merged when style selected)
STYLE_PROMPTS: dict[str, str] = {
    "anime": (
        "Makoto Shinkai style, cel shaded, high-res line art, vibrant, "
        "detailed eyes, 2D only"
    ),
    "realistic": (
        "Photorealistic, 8k UHD, RAW photo, Fujifilm, "
        "highly detailed skin texture, soft lighting"
    ),
    "watercolor": (
        "Traditional watercolor, wet-on-wet, paper texture, paint drips, "
        "soft edges, ethereal"
    ),
    "cyberpunk": (
        "Neon glow, synthwave palette, rain-slicked streets, "
        "high contrast, futuristic, cinematic fog"
    ),
    "oil painting": (
        "Impasto, thick brushstrokes, canvas texture, oil on canvas, "
        "classical art style"
    ),
    "sketch": (
        "Graphite pencil, hand-drawn, charcoal, rough sketch, "
        "white paper, high contrast"
    ),
    "cinematic": (
        "Anamorphic lens, film grain, 35mm film, color graded, "
        "dramatic lighting, depth of field"
    ),
    "fantasy art": (
        "Ethereal, intricate detail, epic composition, "
        "digital illustration, magical atmosphere"
    ),
    "pixel art": (
        "Strictly 2D, flat, 16-bit, SNES style, clean pixels, aliased, "
        "sprite, non-isometric, no depth, side-scroller character, "
        "limited palette, bold black outlines"
    ),
    "3d render": (
        "Octane render, Ray tracing, Unreal Engine 5, 4K, "
        "volumetric lighting, high-poly"
    ),
}

# Style-specific negative keywords (exclude what doesn't fit each style)
STYLE_NEGATIVE_PROMPTS: dict[str, str] = {
    "anime": "3D render, realistic photo, western cartoon, blurry, low quality",
    "realistic": "anime, cartoon, painting, drawing, illustration, stylized",
    "watercolor": "digital art, sharp edges, oil painting, photograph",
    "cyberpunk": "pastel, vintage, natural lighting, daytime, rustic",
    "oil painting": "digital, photo, flat color, minimalist, sketch",
    "sketch": "full color, painted, 3D, photorealistic, polished",
    "cinematic": "flat lighting, amateur, snapshot, overexposed",
    "fantasy art": "realistic, mundane, modern, minimalist",
    "pixel art": (
        "Minecraft, voxel, 3D blocks, lego, isometric 3D, depth, volume, "
        "smooth, anti-aliased, gradients, soft shading, realistic, render"
    ),
    "3d render": "2D, flat, pixel art, painting, drawing, sketch",
}

# Base negative (always applied)
BASE_NEGATIVE = "blurry, low quality, distorted, watermark, text"

DEFAULT_STYLE = "realistic"

# For API /styles: short labels (key -> description)
STYLE_PRESETS: dict[str, str] = {
    "anime": "Anime (Makoto Shinkai, cel shaded, 2D)",
    "realistic": "Realistic (8k UHD, photorealistic)",
    "watercolor": "Watercolor (traditional, soft edges)",
    "cyberpunk": "Cyberpunk (neon, synthwave)",
    "oil painting": "Oil Painting (impasto, classical)",
    "sketch": "Sketch (pencil, charcoal)",
    "cinematic": "Cinematic (35mm, film grain)",
    "fantasy art": "Fantasy Art (ethereal, magical)",
    "pixel art": "Pixel Art (2D flat, SNES style, non-isometric)",
    "3d render": "3D Render (Octane, Unreal Engine 5)",
}


def get_style_prompt(style_key: str) -> str:
    """Return positive prompt template for style."""
    key = style_key.strip().lower()
    return STYLE_PROMPTS.get(key, STYLE_PROMPTS[DEFAULT_STYLE])


def get_style_negative_prompt(style_key: str) -> str:
    """Return negative prompt for style."""
    key = style_key.strip().lower()
    neg = STYLE_NEGATIVE_PROMPTS.get(key, STYLE_NEGATIVE_PROMPTS[DEFAULT_STYLE])
    return f"{BASE_NEGATIVE}, {neg}"


def merge_prompt(style_key: str, custom_prompt: str | None) -> str:
    """Final Prompt = style keywords + user option content."""
    style_text = get_style_prompt(style_key)
    if not (custom_prompt and custom_prompt.strip()):
        return style_text
    return f"{style_text}, {custom_prompt.strip()}"
