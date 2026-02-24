"""
ModelRegistry: one pipeline per Style. No shared model. Read-only after startup.
"""
from __future__ import annotations

import logging
import threading
from pathlib import Path
from typing import TYPE_CHECKING

from app.sdxl_prod.config import get_settings
from app.sdxl_prod.style_enum import Style

if TYPE_CHECKING:
    from diffusers import StableDiffusionXLPipeline

logger = logging.getLogger(__name__)


class ModelRegistry:
    """One pipeline per Style. Same HF path can be loaded multiple times (separate instances)."""

    def __init__(self) -> None:
        self._models: dict[Style, StableDiffusionXLPipeline] = {}

    def register(self, style: Style, model_path: str, lora_path: str | None = None) -> None:
        if style in self._models:
            raise RuntimeError(f"{style!r} already registered")
        import torch
        from diffusers import StableDiffusionXLPipeline

        s = get_settings()
        dtype = torch.float16 if s.torch_dtype == "float16" else torch.float32
        device = s.device

        logger.info("Loading pipeline for %s from %s", style.value, model_path)
        pipe = StableDiffusionXLPipeline.from_pretrained(
            model_path,
            torch_dtype=dtype,
            variant="fp16" if dtype == torch.float16 else None,
            use_safetensors=True,
        ).to(device)

        if not s.safety_checker and hasattr(pipe, "safety_checker"):
            pipe.safety_checker = None
        if s.enable_attention_slicing and hasattr(pipe, "enable_attention_slicing"):
            pipe.enable_attention_slicing()
        if s.enable_vae_slicing and hasattr(pipe, "enable_vae_slicing"):
            pipe.enable_vae_slicing()
        if s.enable_xformers:
            try:
                if hasattr(pipe, "enable_xformers_memory_efficient_attention"):
                    pipe.enable_xformers_memory_efficient_attention()
            except Exception as e:
                logger.warning("xformers not available: %s", e)
        if getattr(__import__("torch").backends.cuda, "matmul", None) is not None and s.enable_tf32:
            torch.backends.cuda.matmul.allow_tf32 = True

        if lora_path:
            try:
                pipe.load_lora_weights(lora_path, adapter_name=f"style_{style.value}")
            except Exception as e:
                logger.warning("LoRA load skipped for %s: %s", style.value, e)
            if hasattr(pipe, "set_adapters"):
                try:
                    adapters = pipe.get_list_adapters()
                    if adapters:
                        pipe.set_adapters(adapters, adapter_weights=[s.lora_default_scale])
                except Exception:
                    pass

        self._models[style] = pipe
        logger.info("Registered %s -> %s", style.value, model_path)

    def get(self, style: Style) -> StableDiffusionXLPipeline:
        if style not in self._models:
            raise RuntimeError(f"Model not found for {style!r}")
        return self._models[style]

    def __contains__(self, style: Style) -> bool:
        return style in self._models


_registry: ModelRegistry | None = None
_registry_lock = threading.Lock()


def get_registry() -> ModelRegistry:
    if _registry is None:
        raise RuntimeError("ModelRegistry not initialized; call initialize_registry() at startup")
    return _registry


def _lora_path(lora_dir: Path, key: str) -> str | None:
    candidates = [
        lora_dir / key / "pytorch_lora_weights.safetensors",
        lora_dir / f"{key}.safetensors",
        lora_dir / key,
    ]
    for p in candidates:
        if p.exists():
            if p.is_dir():
                sf = p / "pytorch_lora_weights.safetensors"
                return str(sf.resolve()) if sf.exists() else str(p.resolve())
            return str(p.resolve())
    return None


def initialize_registry() -> ModelRegistry:
    """Create registry and register one pipeline per style. Same path → separate instances. Call once at startup."""
    global _registry
    with _registry_lock:
        if _registry is not None:
            return _registry
    s = get_settings()
    lora_dir = s.lora_dir

    registry = ModelRegistry()

    registry.register(Style.ANIME, "cagliostrolab/animagine-xl-3.1")
    registry.register(Style.REALISTIC, "SG161222/RealVisXL_V4.0")
    registry.register(Style.WATERCOLOR, "stabilityai/stable-diffusion-xl-base-1.0", _lora_path(lora_dir, "watercolor"))
    registry.register(Style.CYBERPUNK, "Lykon/DreamShaperXL")
    registry.register(Style.OIL, "stabilityai/stable-diffusion-xl-base-1.0", _lora_path(lora_dir, "oil_painting"))
    registry.register(Style.SKETCH, "Lykon/DreamShaperXL")
    registry.register(Style.CINEMATIC, "SG161222/RealVisXL_V4.0")
    registry.register(Style.FANTASY, "Lykon/DreamShaperXL")
    registry.register(Style.PIXEL, "nerijs/pixel-art-xl")
    registry.register(Style.RENDER3D, "stabilityai/stable-diffusion-xl-base-1.0")

    _registry = registry
    return registry
