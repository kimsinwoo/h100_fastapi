"""
Predefined style presets for image-to-image. Each maps to a strong prompt template.
"""

from __future__ import annotations

STYLE_PRESETS: dict[str, str] = {
    "anime": "high quality anime style, studio ghibli inspired, ultra detailed, vibrant colors, clean lines",
    "realistic": "photorealistic, ultra detailed, 8k, professional photography, natural lighting, sharp focus",
    "watercolor": "beautiful watercolor painting, soft edges, flowing colors, artistic brush strokes, delicate",
    "cyberpunk": "cyberpunk style, neon lights, futuristic, dystopian, high contrast, sci-fi atmosphere",
    "oil painting": "classical oil painting, rich textures, impasto, museum quality, fine art, masterpiece",
    "sketch": "detailed pencil sketch, hand drawn, artistic sketch, clean lines, hatching, professional illustration",
    "cinematic": "cinematic lighting, movie still, dramatic composition, anamorphic lens, film grain, blockbuster",
    "fantasy art": "fantasy art, magical, ethereal, detailed illustration, epic, otherworldly, dreamlike",
    "pixel art": (
        "pure 2D pixel art only, flat like a paper drawing, no 3D no depth no volume, "
        "retro 2D game sprite like NES SNES, flat colored pixels on flat background, "
        "sharp black outlines, 8 colors or less, no gradients, no shadows, no anti-aliasing, "
        "sprite sheet style, single flat layer, 2D character art"
    ),
    "3d render": "3D render, octane render, unreal engine, photorealistic 3D, studio lighting, clean render",
}

DEFAULT_STYLE = "realistic"


def get_style_prompt(style_key: str) -> str:
    """Return prompt template for style. Falls back to realistic if unknown."""
    return STYLE_PRESETS.get(style_key.strip().lower(), STYLE_PRESETS[DEFAULT_STYLE])


def merge_prompt(style_key: str, custom_prompt: str | None) -> str:
    """Merge style template with optional custom prompt. Style first, then custom."""
    style_text = get_style_prompt(style_key)
    if not (custom_prompt and custom_prompt.strip()):
        return style_text
    return f"{style_text}, {custom_prompt.strip()}"
