"""
Cloud Theme Rendering Module for z-image-turbo.

GPT CLOUD REPLICA MODE (pet-only): Replicates GPT Cloud theme visual characteristics precisely.
Not generic cloud styling. Environment must dominate; anatomy must stay intact.

Style intensity = HIGH. guidance 9–11, strength 0.6–0.7.
"""

from __future__ import annotations

from typing import Literal

CloudIntensity = Literal["low", "medium", "high"]

# ---------------------------------------------------------------------------
# GPT CLOUD REPLICA — Target characteristics
# ---------------------------------------------------------------------------
# Extremely soft high-key brightness, clean white dominant, very low contrast,
# airy depth, gentle pastel sky-blue diffusion, no dramatic shadow, no grounded realism,
# soft luminous environment, floating spatial softness.

# ----- ENVIRONMENT ENFORCEMENT -----
GPT_CLOUD_ENVIRONMENT = (
    "The pet must exist inside a floating cloud environment. "
    "Entire background composed of bright volumetric clouds. "
    "Clouds must be large, soft, and luminous. "
    "No visible ground plane. No indoor space. No furniture. No realistic landscape. "
    "Background brightness must exceed subject midtones. "
    "Cloud density: medium-high. Cloud softness: maximum. Atmospheric haze: light but visible."
)

# ----- LIGHTING SYSTEM -----
GPT_CLOUD_LIGHTING = (
    "High-key global illumination only. No hard shadows. "
    "Shadow intensity must not exceed soft gray. "
    "Soft ambient wrap lighting. Subtle bloom glow around pet edges. "
    "No directional dramatic light. No cinematic lighting. No contrast spikes."
)

# ----- COLOR SYSTEM -----
GPT_CLOUD_COLOR = (
    "Primary color base: pure white dominant, soft sky blue gradient, very subtle lavender undertone. "
    "No deep blacks. No saturated red or orange dominance. No heavy color grading. "
    "No warm sunset tones. No earth tones. "
    "Overall palette must feel weightless and bright."
)

# ----- PET STRUCTURE PROTECTION -----
GPT_CLOUD_STRUCTURE_PROTECTION = (
    "The pet must remain anatomically accurate. "
    "Preserve: exact pose, exact orientation, visible limbs count, visible eyes count, clothing state if present. "
    "Do NOT: rotate body, add symmetry correction, convert pose to frontal, add fantasy distortion. "
    "Cloud style must NOT distort anatomy."
)

# ----- RENDERING TEXTURE -----
GPT_CLOUD_RENDERING = (
    "Fur must remain detailed but softly lit. Edge transitions must be smooth. "
    "No sharp contrast edges. No gritty detail. No heavy sharpening. No texture exaggeration. "
    "Surface finish: clean, modern, slightly soft finish. "
    "Not painterly. Not oil painting. Not clay. Not cartoonish."
)

# ----- STYLE DOMINANCE RULE -----
GPT_CLOUD_DOMINANCE = (
    "Cloud atmosphere must be visually dominant. "
    "If the result looks like a normal photo with a sky background, regenerate. "
    "If the result contains ground or realistic scene depth, regenerate. "
    "If contrast feels cinematic or dramatic, regenerate."
)

# ----- NEGATIVE PROMPT BLOCK (GPT Cloud Replica) -----
GPT_CLOUD_NEGATIVE = (
    "cinematic lighting, dramatic shadow, studio background, solid background, "
    "indoor, grass, field, wood, concrete, sunset, night, dark sky, "
    "strong contrast, moody atmosphere, film grain, high saturation, "
    "earth tone, warm orange, harsh light, "
    "dark dramatic lighting, deep black shadows, low exposure, "
    "realistic ground plane, furniture, landscape, grounded realism"
)

# ---------------------------------------------------------------------------
# GPT CLOUD PHOTOREAL MODE — Real photograph in real sky. NOT illustrated/stylized.
# ---------------------------------------------------------------------------
# Result must remain photorealistic. Professional outdoor pet photoshoot on bright cloudy day.

# ----- PHOTOREAL ENFORCEMENT -----
GPT_CLOUD_PHOTOREAL_ENFORCEMENT = (
    "The pet must look like a real photograph. Natural fur texture. Realistic depth of field. "
    "True-to-life proportions. No painterly effect. No cartoon rendering. No clay texture. "
    "No 3D stylization. No digital art look."
)

# ----- CLOUD ENVIRONMENT (REAL SKY MODE) -----
GPT_CLOUD_PHOTOREAL_ENVIRONMENT = (
    "The pet must appear photographed in a bright cloud-filled sky environment. "
    "Realistic sky with soft white clouds. Natural atmospheric perspective. Real sky lighting behavior. "
    "No fantasy floating effect. No surreal glow. Clouds must look like real sky clouds, not illustrated clouds."
)

# ----- LIGHTING -----
GPT_CLOUD_PHOTOREAL_LIGHTING = (
    "Soft daylight. High-key exposure. Natural global illumination. "
    "No bloom exaggeration. No glowing outlines. No soft fantasy haze. No ethereal glow. "
    "Shadow behavior: very soft, physically correct. No dramatic cinematic contrast."
)

# ----- COLOR PROFILE -----
GPT_CLOUD_PHOTOREAL_COLOR = (
    "Clean whites. Soft sky blues. Natural fur color preserved. "
    "No lavender tint. No pastel wash. No dreamy filter. "
    "Professional outdoor pet photoshoot on a bright cloudy day."
)

# ----- STRUCTURE LOCK (same as replica) -----
GPT_CLOUD_PHOTOREAL_STRUCTURE = (
    "Preserve pose. Preserve visible limbs. Preserve anatomy. Preserve clothing state. "
    "No rotation. No frontal correction."
)

# ----- NEGATIVE PROMPT (STRICT) -----
GPT_CLOUD_PHOTOREAL_NEGATIVE = (
    "illustration, painting, digital art, anime, cartoon, 3D render, clay render, "
    "oil paint, watercolor, soft fantasy glow, ethereal light, surreal, stylized, "
    "artistic filter, overprocessed, hdr dramatic, "
    "cinematic lighting, dramatic shadow, moody atmosphere, low key, deep black shadows"
)

# ---------------------------------------------------------------------------
# Legacy / fallback blocks (used when intensity is low or medium)
# ---------------------------------------------------------------------------

CLOUD_THEME_ENVIRONMENT = (
    "The entire scene must exist inside a cloud world. "
    "Fully bright sky. Large volumetric clouds filling 70% of frame. "
    "No ground texture. No indoor environment. No dark background. No neutral background. "
    "Sky and clouds dominate the scene. Airy cloudscape environment."
)

CLOUD_THEME_LIGHTING = (
    "High-key lighting only. Scene brightness must be elevated. "
    "No shadows darker than soft gray. Global diffuse illumination. "
    "Atmospheric light scattering. Soft bloom effect. Subtle haze depth. "
    "No dramatic lighting. No cinematic lighting."
)

CLOUD_THEME_COLOR = (
    "Primary palette: white, soft sky blue, very light lavender. "
    "No deep black. No strong contrast. No dark gradients. No muted earthy tones. "
    "Bright airy color only."
)

CLOUD_THEME_TEXTURE = (
    "Smooth shading. Soft edge blending. Subtle glow. High resolution. "
    "Clean modern aesthetic. No grain. No noise. No gritty texture. "
    "Airy soft rendering, not realistic-grounded."
)

CLOUD_THEME_STRUCTURE_PROTECTION = (
    "Preserve exact pose. Preserve visible limbs. Preserve visible eyes. Preserve orientation. "
    "No pose correction. No symmetry correction. No perspective normalization. No anatomy alteration. "
    "Subject anatomy must remain intact; only environment and lighting are cloud-themed."
)

CLOUD_VISUAL_CHECK_RULE_DOC = (
    "If the background does not clearly show volumetric clouds, regenerate. "
    "If the lighting is dramatic or cinematic, regenerate. "
    "If the scene looks realistic-grounded instead of airy, regenerate."
)

CLOUD_THEME_NEGATIVE = (
    "dark dramatic lighting, strong contrast, cinematic shadows, moody atmosphere, "
    "low exposure, deep black shadows, grain, film noise, heavy texture, over sharpened, "
    "oversaturated colors, overexposure, harsh shadows, contrast distortion, "
    "indoor, studio background, solid color background, dark sky, night scene, "
    "sunset orange sky, moody lighting, cinematic contrast, dramatic shadows, "
    "earth tones, grass field, wood texture, concrete, realistic ground plane"
)

# ---------------------------------------------------------------------------
# Validation loop criteria (for scene classifier / future use)
# ---------------------------------------------------------------------------
# After generation, check: (1) Background fully cloud-dominant?
# (2) Lighting high-key and low contrast? (3) Pet anatomy unchanged?
# (4) No dark shadows? (5) Scene feels airy and weightless?
# If any fail → regenerate automatically.
GPT_CLOUD_VALIDATION_CRITERIA = (
    "1. Is the background fully cloud-dominant? "
    "2. Is lighting high-key and low contrast? "
    "3. Is the pet anatomy unchanged? "
    "4. Are there no dark shadows? "
    "5. Does the scene feel airy and weightless?"
)


def get_gpt_cloud_replica_block() -> str:
    """
    Full GPT Cloud Replica block for pet-only generation.
    Replicates GPT Cloud theme precisely: environment + lighting + color + structure + rendering + dominance.
    """
    return (
        f"{GPT_CLOUD_STRUCTURE_PROTECTION} "
        f"{GPT_CLOUD_ENVIRONMENT} {GPT_CLOUD_LIGHTING} {GPT_CLOUD_COLOR} "
        f"{GPT_CLOUD_RENDERING} {GPT_CLOUD_DOMINANCE}"
    )


def get_gpt_cloud_replica_negative() -> str:
    """Negative prompt for GPT Cloud Replica Mode. Use with pose-lock negative."""
    return GPT_CLOUD_NEGATIVE


def get_gpt_cloud_photoreal_block() -> str:
    """
    Full GPT Cloud Photoreal block: real photograph in real sky.
    Photorealistic only. No illustrated, painted, or stylized look.
    """
    return (
        f"{GPT_CLOUD_PHOTOREAL_STRUCTURE} {GPT_CLOUD_PHOTOREAL_ENFORCEMENT} "
        f"{GPT_CLOUD_PHOTOREAL_ENVIRONMENT} {GPT_CLOUD_PHOTOREAL_LIGHTING} "
        f"{GPT_CLOUD_PHOTOREAL_COLOR}"
    )


def get_gpt_cloud_photoreal_negative() -> str:
    """Negative prompt for GPT Cloud Photoreal Mode. Strict anti-illustration."""
    return GPT_CLOUD_PHOTOREAL_NEGATIVE


def get_cloud_theme_style_block(intensity: CloudIntensity = "high") -> str:
    """
    Returns cloud style block. intensity='high' → GPT Cloud Replica (pet-only precise).
    Low/medium = softer legacy blocks.
    """
    intensity = (intensity or "high").strip().lower()
    if intensity not in ("low", "medium", "high"):
        intensity = "high"

    if intensity == "high":
        return (
            f"{GPT_CLOUD_ENVIRONMENT} {GPT_CLOUD_LIGHTING} {GPT_CLOUD_COLOR} "
            f"{GPT_CLOUD_RENDERING} {GPT_CLOUD_DOMINANCE}"
        )
    if intensity == "low":
        return (
            "Soft cloud atmosphere: bright pastel sky, light haze, gentle diffused light. "
            "High-key, no harsh shadows, elevated brightness. "
            "Smooth shading, clean edges, no grain. White and soft blue dominant."
        )
    return (
        f"{CLOUD_THEME_ENVIRONMENT} {CLOUD_THEME_LIGHTING} "
        f"{CLOUD_THEME_COLOR} {CLOUD_THEME_TEXTURE}"
    )


def get_cloud_theme_block(intensity: CloudIntensity = "high") -> str:
    """
    Full cloud theme block: structure protection + style.
    intensity='high' → GPT Cloud Replica (structure + full style).
    """
    intensity = (intensity or "high").strip().lower()
    if intensity not in ("low", "medium", "high"):
        intensity = "high"
    if intensity == "high":
        return get_gpt_cloud_replica_block()
    return (
        f"{CLOUD_THEME_STRUCTURE_PROTECTION} "
        f"{get_cloud_theme_style_block(intensity)}"
    )


def inject_cloud_theme_into_prompt(
    base_prompt: str,
    intensity: CloudIntensity = "high",
) -> str:
    """Appends cloud theme after base_prompt. Pose-lock stays first."""
    base = (base_prompt or "").strip()
    if intensity == "high":
        structure = GPT_CLOUD_STRUCTURE_PROTECTION
        style = get_cloud_theme_style_block("high")
    else:
        structure = CLOUD_THEME_STRUCTURE_PROTECTION
        style = get_cloud_theme_style_block(intensity)
    if not base:
        return f"{structure} {style}"
    return f"{base} {structure} {style}"


def get_cloud_theme_negative(use_gpt_replica: bool = True) -> str:
    """
    Negative prompt for cloud theme.
    use_gpt_replica=True (default): GPT Cloud Replica block for pet-only.
    """
    return GPT_CLOUD_NEGATIVE if use_gpt_replica else CLOUD_THEME_NEGATIVE
