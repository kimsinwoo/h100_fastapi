"""
Prompt engine using STYLE_REGISTRY.
Injects style-specific positive/negative templates and resolves effective cfg/steps/strength/sampler.
If client does not send cfg/steps/strength, style defaults are used; otherwise user values are respected (clamped to global limits).
"""
from __future__ import annotations

from typing import Tuple

from app.sd15.config import Settings
from app.sd15.config import get_settings
from app.sd15.style_registry import StyleConfig
from app.sd15.style_registry import get_style_config

SUBJECT_PLACEHOLDER = "{subject}"


def build_positive_prompt(user_prompt: str, style_key: str) -> str:
    """Inject style-specific positive template; {subject} is replaced by user content."""
    config = get_style_config(style_key)
    template = config["positive_prompt_template"]
    subject = user_prompt.strip() if user_prompt else "subject"
    if SUBJECT_PLACEHOLDER not in template:
        return f"{template}, {subject}"
    return template.replace(SUBJECT_PLACEHOLDER, subject)


def build_negative_prompt(style_key: str) -> str:
    """Return style-specific negative template."""
    config = get_style_config(style_key)
    return config["negative_prompt_template"]


def get_effective_params(
    style_key: str,
    user_cfg: float | None,
    user_steps: int | None,
    user_strength: float | None,
    settings: Settings | None = None,
) -> Tuple[float, int, float, str]:
    """
    Resolve effective cfg, steps, strength, sampler.
    If user did not send a value, use style default; otherwise clamp user value to global limits.
    Strength is further clamped to style's recommended_strength_range.
    Returns: (cfg, steps, strength, sampler_name).
    """
    s = settings or get_settings()
    config = get_style_config(style_key)
    rec = config["recommended_strength_range"]
    str_min, str_max = rec["min"], rec["max"]

    cfg: float = (
        float(config["default_cfg"])
        if user_cfg is None
        else max(1.0, min(20.0, user_cfg))
    )
    steps: int = (
        config["default_steps"]
        if user_steps is None
        else max(1, min(100, user_steps))
    )
    strength: float = (
        max(str_min, min(str_max, s.default_strength))
        if user_strength is None
        else max(str_min, min(str_max, max(0.0, min(1.0, user_strength))))
    )
    sampler: str = config["default_sampler"]
    return (cfg, steps, strength, sampler)


def style_enable_upscale(style_key: str, user_upscale: bool | None) -> bool:
    """If user explicitly set upscale, respect it; else use style's enable_upscale (e.g. Pixel Art disables)."""
    config = get_style_config(style_key)
    if user_upscale is not None:
        return user_upscale
    return config["enable_upscale"]


def style_output_grayscale(style_key: str) -> bool:
    """Whether to convert output to grayscale (e.g. Sketch)."""
    config = get_style_config(style_key)
    return config["output_grayscale"]
