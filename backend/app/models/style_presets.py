"""
Style display names for API (/styles). No prompt injection.
Prompt building is in app.utils.prompt_builder.
"""

from __future__ import annotations

from app.utils.prompt_builder import STYLE_PROMPTS

STYLE_PRESETS: dict[str, str] = {
    "sailor_moon": "Sailor Moon (magical girl, sparkle)",
    "pixel_art": "Pixel Art (sprite, 16 colors)",
    "animal_crossing": "게임 캐릭터 (구조 재디자인)",
    "clay_art": "클레이 아트 (손수 제작 점토 조각)",
}


def get_style_prompt(style_key: str) -> str:
    """Style construction rules (for LLM suggestion). No base injection."""
    key = style_key.strip().lower()
    key_no_space = key.replace(" ", "_")
    return STYLE_PROMPTS.get(key) or STYLE_PROMPTS.get(key_no_space) or key
