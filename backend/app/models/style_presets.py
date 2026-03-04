"""
Style display names for API (/styles). No prompt injection.
Prompt building is in app.utils.prompt_builder.
"""

from __future__ import annotations

from app.utils.prompt_builder import STYLE_PROMPTS

STYLE_PRESETS: dict[str, str] = {
    "dragonball": "Dragon Ball (2-tone cel, thick outline)",
    "slamdunk": "Slam Dunk (sports manga, angular)",
    "sailor_moon": "Sailor Moon (magical girl, sparkle)",
    "pokemon": "Pokemon (cute creature, flat color)",
    "dooly": "Dooly (simple comic strip)",
    "mazinger": "Mazinger (super robot, mecha)",
    "shinchan": "Shinchan (gag manga)",
    "pixel_art": "Pixel Art (sprite, 16 colors)",
    "pixel art": "Pixel Art (sprite, 16 colors)",
}


def get_style_prompt(style_key: str) -> str:
    """Style construction rules (for LLM suggestion). No base injection."""
    key = style_key.strip().lower()
    key_no_space = key.replace(" ", "_")
    return STYLE_PROMPTS.get(key) or STYLE_PROMPTS.get(key_no_space) or key
