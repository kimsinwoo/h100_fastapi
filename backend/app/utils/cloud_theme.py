"""
Cloud Theme Rendering Module for z-image-turbo.

Ultra-soft, high-key cloud-themed images while preserving subject structure.
Modular and injectable into existing pose-lock systems. Never overrides pose,
orientation, or symmetry. Configurable intensity: low / medium / high.
"""

from __future__ import annotations

from typing import Literal

CloudIntensity = Literal["low", "medium", "high"]

# ---------------------------------------------------------------------------
# Style block components (environment, lighting, color, texture)
# ---------------------------------------------------------------------------

CLOUD_THEME_ENVIRONMENT = (
    "Bright pastel sky background, fluffy volumetric clouds, soft atmospheric haze, "
    "floating subtle light particles."
)

CLOUD_THEME_LIGHTING = (
    "High-key exposure, soft diffused global illumination, no harsh shadows, no dramatic contrast, "
    "soft rim light from clouds, gentle ambient bounce light."
)

CLOUD_THEME_COLOR = (
    "Dominant white, soft sky blue, subtle lavender highlights, no oversaturation, no deep blacks."
)

CLOUD_THEME_TEXTURE = (
    "Smooth shading transitions, soft edge blending, subtle glow effect, high resolution detail, "
    "clean modern aesthetic, no grain, no noise, no gritty texture."
)

# Structure protection: always included so cloud never overrides pose-lock
CLOUD_THEME_STRUCTURE_PROTECTION = (
    "Preserve exact pose. Preserve visible limbs. Preserve visible eyes. Preserve orientation. "
    "No pose correction. No symmetry correction. No perspective normalization. No anatomy alteration."
)

# Negative block
CLOUD_THEME_NEGATIVE = (
    "dark dramatic lighting, strong contrast, cinematic shadows, moody atmosphere, "
    "low exposure, deep black shadows, grain, film noise, heavy texture, over sharpened, "
    "oversaturated colors, overexposure, harsh shadows, contrast distortion"
)


def get_cloud_theme_style_block(intensity: CloudIntensity = "medium") -> str:
    """
    Returns cloud style only (environment + lighting + color + texture).
    Intensity: low = subtle, medium = balanced, high = strong dreamy effect without structural distortion.
    """
    intensity = (intensity or "medium").strip().lower()
    if intensity not in ("low", "medium", "high"):
        intensity = "medium"

    if intensity == "low":
        return (
            "Subtle cloud atmosphere: soft pastel sky, light haze, gentle diffused light. "
            "High-key, no harsh shadows, balanced brightness. "
            "Smooth shading, clean edges, no grain, no oversaturation."
        )
    if intensity == "high":
        return (
            f"{CLOUD_THEME_ENVIRONMENT} {CLOUD_THEME_LIGHTING} {CLOUD_THEME_COLOR} {CLOUD_THEME_TEXTURE} "
            "Dreamy ethereal cloudscape, strong soft cloud presence, luminous atmosphere. "
            "No structural distortion, no anatomy alteration."
        )
    # medium (default balanced)
    return (
        f"{CLOUD_THEME_ENVIRONMENT} {CLOUD_THEME_LIGHTING} "
        f"{CLOUD_THEME_COLOR} {CLOUD_THEME_TEXTURE}"
    )


def get_cloud_theme_block(intensity: CloudIntensity = "medium") -> str:
    """
    Full cloud theme block for standalone use: structure protection + style.
    Use when cloud_theme is the selected style (not injected into pose-lock).
    """
    return (
        f"{CLOUD_THEME_STRUCTURE_PROTECTION} "
        f"{get_cloud_theme_style_block(intensity)}"
    )


def inject_cloud_theme_into_prompt(
    base_prompt: str,
    intensity: CloudIntensity = "medium",
) -> str:
    """
    Safe merge: appends cloud theme after base_prompt. Cloud style never overrides
    pose-lock instructions (base_prompt stays first and intact).
    Use for pose-lock systems: pass pose-lock prompt as base_prompt, get combined prompt.
    """
    base = (base_prompt or "").strip()
    cloud_block = get_cloud_theme_style_block(intensity)
    structure = CLOUD_THEME_STRUCTURE_PROTECTION
    # Append structure + cloud so they reinforce; base (pose-lock) is never modified
    if not base:
        return f"{structure} {cloud_block}"
    return f"{base} {structure} {cloud_block}"


def get_cloud_theme_negative() -> str:
    """Negative prompt block for cloud theme. Use in merge with other negatives if needed."""
    return CLOUD_THEME_NEGATIVE
