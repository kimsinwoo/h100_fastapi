"""
Style display names for API (/styles). No prompt injection.
Prompt building is in app.utils.prompt_builder.
"""

from __future__ import annotations

from app.utils.prompt_builder import STYLE_TEMPLATES

STYLE_PRESETS: dict[str, str] = {
    "anime": "Anime (2D Cel-Shaded)",
    "realistic": "Realistic (Photographic)",
    "watercolor": "Watercolor (Traditional)",
    "cyberpunk": "Cyberpunk (Futuristic)",
    "oil painting": "Oil Painting (Classical)",
    "sketch": "Sketch (Hand-drawn)",
    "cinematic": "Cinematic (Movie Scene)",
    "fantasy art": "Fantasy Art (Digital Illustration)",
    "pixel art": "Pixel Art (2D Flat Sprite)",
    "3d render": "3D Render (CGI)",
    "omni": "Omni (Fotorealistic / High Detail)",
}


def get_style_prompt(style_key: str) -> str:
    """Minimal style description (for LLM suggestion). No base injection."""
    key = style_key.strip().lower()
    key_no_space = key.replace(" ", "_")
    return STYLE_TEMPLATES.get(key) or STYLE_TEMPLATES.get(key_no_space) or key
