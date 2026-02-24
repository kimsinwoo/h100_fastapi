"""
Model keys and HuggingFace IDs for style → model mapping.
Each style uses its optimal dedicated model; no single shared base for all.
"""
from __future__ import annotations

# Model keys (internal); HF IDs are in config or here for defaults
MODEL_KEY_ANIMAGINE_XL = "animagine_xl"
MODEL_KEY_REALVIS_XL = "realvis_xl"
MODEL_KEY_JUGGERNAUT_XL = "juggernaut_xl"
MODEL_KEY_DREAMSHAPER_XL = "dreamshaper_xl"
MODEL_KEY_SDXL_BASE = "sdxl_base"
MODEL_KEY_PIXELART_XL = "pixelart_xl"

DEFAULT_MODEL_IDS: dict[str, str] = {
    MODEL_KEY_ANIMAGINE_XL: "cagliostrolab/animagine-xl-3.0",
    MODEL_KEY_REALVIS_XL: "SG161222/RealVisXL_V4.0",
    MODEL_KEY_JUGGERNAUT_XL: "juggernautxl/juggernaut-xl-v8",
    MODEL_KEY_DREAMSHAPER_XL: "Lykon/dreamshaper-xl-1-0",
    MODEL_KEY_SDXL_BASE: "stabilityai/stable-diffusion-xl-base-1.0",
    MODEL_KEY_PIXELART_XL: "ptx0/pixel-art-xl",
}

ALL_MODEL_KEYS: tuple[str, ...] = (
    MODEL_KEY_ANIMAGINE_XL,
    MODEL_KEY_REALVIS_XL,
    MODEL_KEY_JUGGERNAUT_XL,
    MODEL_KEY_DREAMSHAPER_XL,
    MODEL_KEY_SDXL_BASE,
    MODEL_KEY_PIXELART_XL,
)
