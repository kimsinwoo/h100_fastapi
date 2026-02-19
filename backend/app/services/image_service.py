"""
Production-grade Z-Image-Turbo img2img service
Optimized for quality, stability, and commercial deployment
"""

from __future__ import annotations

import asyncio
import io
import logging
import time
import warnings
from typing import Any

from app.core.config import get_settings
from app.models.style_presets import merge_prompt

logger = logging.getLogger(__name__)

_pipeline: Any = None
_lock = asyncio.Lock()
_device: Any = None

# ===== Z-Image-Turbo 권장값 (스케줄러는 기본 FlowMatchEulerDiscrete 사용) =====
DEFAULT_STRENGTH = 0.65
DEFAULT_GUIDANCE_SCALE = 0.0
DEFAULT_NUM_INFERENCE_STEPS = 8
MODEL_RESOLUTION = 1024

# Pure 2D pixel art: preserve pose, convert 3D→2D, 8 colors, black outlines, no AA
PIXEL_POSITIVE_ADD = (
    "strictly 2D only, flat 2D pixels, orthographic view, "
    "preserve exact same shape and pose, convert voxel to flat pixel sprite, "
    "maximum 8 colors, limited palette, bold black outlines, "
    "pixel-perfect, sharp edges, no anti-aliasing, no gradients, no soft shading, "
    "retro 8-bit 16-bit sprite, blocky, clean, minimal shading"
)

PIXEL_NEGATIVE_ADD = (
    "3D, voxel, lego, cube, volume, depth, perspective, "
    "ray tracing, render, realism, photorealistic, "
    "anti-aliasing, soft edges, gradients, smooth shading, blur, "
    "depth of field, volumetric lighting, more than 8 colors, complex palette"
)


# ============================================================
# Device
# ============================================================

def _resolve_device():
    import torch
    if torch.cuda.is_available():
        return torch.device("cuda")
    if getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


# ============================================================
# Load Pipeline
# ============================================================

def _load_pipeline_sync():
    global _pipeline, _device

    import torch
    from diffusers import ZImageImg2ImgPipeline

    settings = get_settings()
    _device = _resolve_device()

    dtype = torch.bfloat16 if (_device.type == "cuda" and getattr(torch, "bfloat16", None)) else torch.float32

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")

        pipe = ZImageImg2ImgPipeline.from_pretrained(
            settings.model_id,
            torch_dtype=dtype,
            low_cpu_mem_usage=True,
        )
        # Z-Image 전용 스케줄러 유지 (UniPCMultistepScheduler 교체 시 set_timesteps AssertionError 발생)

        for method_name in ("enable_attention_slicing", "enable_vae_slicing", "enable_vae_tiling"):
            method = getattr(pipe, method_name, None)
            if callable(method):
                try:
                    method()
                except Exception:
                    pass

        pipe = pipe.to(_device)
        # VAE는 파이프라인과 동일 dtype 유지 (float32로 바꾸면 Half/float 불일치로 오류)

    logger.info(
        "Pipeline loaded on %s (dtype=%s)",
        _device,
        dtype,
    )

    _pipeline = pipe
    return pipe


async def get_pipeline():
    global _pipeline
    async with _lock:
        if _pipeline is None:
            loop = asyncio.get_event_loop()
            _pipeline = await loop.run_in_executor(
                None, _load_pipeline_sync
            )
        return _pipeline


# ============================================================
# Inference
# ============================================================

def _run_inference_sync(
    image_bytes: bytes,
    prompt: str,
    negative_prompt: str,
    strength: float,
    num_steps: int,
    guidance_scale: float,
    size: int,
    seed: int | None,
) -> bytes:

    import torch
    from PIL import Image

    global _pipeline, _device

    if _pipeline is None:
        raise RuntimeError("Pipeline not loaded")

    # Load image
    img = Image.open(io.BytesIO(image_bytes)).convert("RGB")
    img = img.resize((size, size), Image.Resampling.LANCZOS)

    # Deterministic seed
    generator = torch.Generator(device=_device)
    if seed is not None:
        generator.manual_seed(seed)

    logger.info(
        "Running img2img | steps=%d | guidance=%.2f | strength=%.2f",
        num_steps,
        guidance_scale,
        strength,
    )

    with torch.inference_mode():
        result = _pipeline(
            prompt=prompt,
            negative_prompt=negative_prompt,
            image=img,
            strength=strength,
            num_inference_steps=num_steps,
            guidance_scale=guidance_scale,
            generator=generator,
            output_type="pil",   # 🔥 Let diffusers handle decoding
        )

    image = result.images[0]

    buf = io.BytesIO()
    image.save(buf, format="PNG")
    return buf.getvalue()


# ============================================================
# Public API
# ============================================================

async def run_image_to_image(
    image_bytes: bytes,
    style_key: str,
    custom_prompt: str | None = None,
    strength: float | None = None,
    num_steps: int | None = None,
    size: int | None = None,
    seed: int | None = None,
):

    pipe = await get_pipeline()
    if pipe is None:
        raise RuntimeError("Model not available")

    settings = get_settings()

    prompt = merge_prompt(style_key, custom_prompt)

    negative_prompt = (
        "blurry, low quality, distorted, watermark, text"
    )

    # Pixel preset: enforce 2D sprite, preserve pose, 8 colors, black outlines
    if "pixel" in style_key.lower():
        prompt = f"{prompt}, {PIXEL_POSITIVE_ADD}"
        negative_prompt = f"{negative_prompt}, {PIXEL_NEGATIVE_ADD}"
        # Slightly higher strength to convert 3D voxel → 2D pixel while keeping pose
        strength = strength or 0.72
    else:
        strength = strength or DEFAULT_STRENGTH

    strength = max(0.0, min(1.0, strength))
    num_steps = max(1, min(50, num_steps or DEFAULT_NUM_INFERENCE_STEPS))
    guidance_scale = DEFAULT_GUIDANCE_SCALE

    resolution = size or MODEL_RESOLUTION

    loop = asyncio.get_event_loop()
    start = time.perf_counter()

    result = await loop.run_in_executor(
        None,
        lambda: _run_inference_sync(
            image_bytes,
            prompt,
            negative_prompt,
            strength,
            num_steps,
            guidance_scale,
            resolution,
            seed,
        ),
    )

    elapsed = time.perf_counter() - start
    return result, elapsed


# ============================================================
# Utilities
# ============================================================

def is_pipeline_loaded() -> bool:
    return _pipeline is not None


def is_gpu_available() -> bool:
    try:
        import torch
        return torch.cuda.is_available()
    except Exception:
        return False
