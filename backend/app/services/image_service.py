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

# ===== High Quality Defaults =====
DEFAULT_STRENGTH = 0.65
DEFAULT_GUIDANCE_SCALE = 7.5
DEFAULT_NUM_INFERENCE_STEPS = 35
MODEL_RESOLUTION = 1024

PIXEL_POSITIVE_ADD = (
    "strictly 2D, flat shading, hard pixel edges, "
    "limited color palette, no gradients, orthographic view"
)

PIXEL_NEGATIVE_ADD = (
    "3D, voxel, lego, render, ray tracing, "
    "depth of field, volumetric lighting"
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
    from diffusers import ZImageImg2ImgPipeline, UniPCMultistepScheduler

    settings = get_settings()
    _device = _resolve_device()

    dtype = torch.bfloat16 if _device.type == "cuda" else torch.float32

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")

        pipe = ZImageImg2ImgPipeline.from_pretrained(
            settings.model_id,
            torch_dtype=dtype,
            low_cpu_mem_usage=True,
        )

        # 🔥 Replace scheduler (huge quality boost)
        pipe.scheduler = UniPCMultistepScheduler.from_config(
            pipe.scheduler.config
        )

        # Memory optimizations
        if hasattr(pipe, "enable_attention_slicing"):
            pipe.enable_attention_slicing()
        if hasattr(pipe, "enable_vae_slicing"):
            pipe.enable_vae_slicing()
        if hasattr(pipe, "enable_vae_tiling"):
            pipe.enable_vae_tiling()

        pipe = pipe.to(_device)

        # 🔥 Critical: Keep VAE in float32 for stability
        if hasattr(pipe, "vae"):
            pipe.vae.to(dtype=torch.float32)

    logger.info(
        "Pipeline loaded on %s (main dtype=%s, VAE=float32)",
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

    # Pixel preset
    if "pixel" in style_key.lower():
        prompt = f"{prompt}, {PIXEL_POSITIVE_ADD}"
        negative_prompt = f"{negative_prompt}, {PIXEL_NEGATIVE_ADD}"

    strength = max(0.0, min(1.0, strength or DEFAULT_STRENGTH))
    num_steps = max(10, min(60, num_steps or DEFAULT_NUM_INFERENCE_STEPS))
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
