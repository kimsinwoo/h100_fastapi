"""
ModelRegistry: 단일 책임. Style당 전용 파이프라인. freeze 후 read-only.
전역 단일 pipeline 없음. 모델 로딩은 서버 시작 시에만 발생.
"""
from __future__ import annotations

import logging
import sys
import threading
from typing import TYPE_CHECKING

import torch

from app.sdxl_prod.config import get_settings
from app.sdxl_prod.style_enum import Style

if TYPE_CHECKING:
    from diffusers import StableDiffusionXLPipeline

logger = logging.getLogger(__name__)


class ModelRegistry:
    """Style당 하나의 파이프라인. freeze 이후 수정 불가. Fallback 없음."""

    def __init__(self) -> None:
        self._models: dict[Style, StableDiffusionXLPipeline] = {}
        self._frozen = False

    def register(self, style: Style, path: str) -> None:
        if self._frozen:
            raise RuntimeError("Registry already frozen")
        if style in self._models:
            raise RuntimeError(f"{style!r} already registered")

        from diffusers import StableDiffusionXLPipeline

        s = get_settings()
        device = s.device
        dtype = torch.float16

        logger.info("Loading pipeline for style=%s path=%s", style.value, path)
        pipe = StableDiffusionXLPipeline.from_pretrained(
            path,
            torch_dtype=dtype,
            variant="fp16",
            use_safetensors=True,
        ).to(device)

        if not s.safety_checker and hasattr(pipe, "safety_checker"):
            pipe.safety_checker = None

        if hasattr(pipe, "vae") and pipe.vae is not None and hasattr(pipe.vae, "enable_slicing"):
            pipe.vae.enable_slicing()

        if hasattr(pipe, "enable_xformers_memory_efficient_attention"):
            try:
                pipe.enable_xformers_memory_efficient_attention()
            except Exception as e:
                logger.warning("xformers not available: %s", e)

        if getattr(torch.backends.cuda, "matmul", None) is not None and s.enable_tf32:
            torch.backends.cuda.matmul.allow_tf32 = True

        self._models[style] = pipe
        model_name = getattr(getattr(pipe, "config", None), "_name_or_path", path)
        print(f"[REGISTRY] style={style.value} 로드된 모델={model_name}", file=sys.stderr, flush=True)

    def freeze(self) -> None:
        self._frozen = True

    def get(self, style: Style) -> StableDiffusionXLPipeline:
        if style not in self._models:
            raise RuntimeError(f"Model not registered: {style}")
        return self._models[style]

    def __contains__(self, style: Style) -> bool:
        return style in self._models


_registry: ModelRegistry | None = None
_registry_lock = threading.Lock()


def get_registry() -> ModelRegistry:
    if _registry is None:
        raise RuntimeError("ModelRegistry not initialized; call initialize_registry() at startup")
    return _registry


def initialize_registry() -> ModelRegistry:
    """서버 시작 시 1회만 호출. 모든 스타일 preload. 이후 register 금지."""
    global _registry
    with _registry_lock:
        if _registry is not None:
            return _registry

    registry = ModelRegistry()

    registry.register(Style.PIXEL, "nerijs/pixel-art-xl")
    registry.register(Style.CYBERPUNK, "Lykon/DreamShaperXL")
    registry.register(Style.REALISTIC, "SG161222/RealVisXL_V4.0")
    registry.register(Style.ANIME, "cagliostrolab/animagine-xl-3.1")
    registry.register(Style.FANTASY, "Lykon/DreamShaperXL")
    registry.register(Style.CINEMATIC, "SG161222/RealVisXL_V4.0")
    registry.register(Style.WATERCOLOR, "stabilityai/stable-diffusion-xl-base-1.0")
    registry.register(Style.OIL, "stabilityai/stable-diffusion-xl-base-1.0")
    registry.register(Style.SKETCH, "Lykon/DreamShaperXL")
    registry.register(Style.RENDER3D, "stabilityai/stable-diffusion-xl-base-1.0")

    registry.freeze()
    _registry = registry
    return registry
