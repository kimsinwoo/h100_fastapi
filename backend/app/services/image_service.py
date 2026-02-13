"""
Z-Image-Turbo image-to-image service. Singleton pipeline, loaded once at startup.
GPU if available with autocast, fallback to CPU. Memory optimized.
"""

from __future__ import annotations

import asyncio
import io
import logging
import time
from typing import Any

from app.core.config import get_settings
from app.models.style_presets import merge_prompt

logger = logging.getLogger(__name__)

_pipeline: Any = None
_lock = asyncio.Lock()
_device: Any = None
_use_autocast: bool = True


def _resolve_device() -> Any:
    import torch
    settings = get_settings()
    if settings.device_preference == "cpu":
        return torch.device("cpu")
    if settings.device_preference == "cuda" and torch.cuda.is_available():
        return torch.device("cuda")
    if settings.device_preference == "auto":
        if torch.cuda.is_available():
            return torch.device("cuda")
        if getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
            return torch.device("mps")
    return torch.device("cpu")


def _load_pipeline_sync() -> Any:
    """Load Z-Image-Turbo pipeline once. Called at startup."""
    global _pipeline, _device, _use_autocast
    try:
        import torch
    except ImportError as e:
        logger.error("torch not installed: %s", e)
        return None
    try:
        from diffusers import ZImageImg2ImgPipeline
    except ImportError as e:
        logger.error("ZImageImg2ImgPipeline not available (diffusers>=0.36): %s", e)
        return None
    settings = get_settings()
    _device = _resolve_device()
    _use_autocast = settings.use_autocast and _device.type in ("cuda", "mps")
    dtype = torch.float16
    if _device.type == "cuda" and hasattr(torch, "bfloat16"):
        dtype = torch.bfloat16
    if _device.type == "cpu":
        dtype = torch.float32
    try:
        pipe = ZImageImg2ImgPipeline.from_pretrained(
            settings.model_id,
            torch_dtype=dtype,
            low_cpu_mem_usage=True,
        )
        pipe.set_progress_bar_config(disable=True)
        try:
            pipe.enable_attention_slicing()
        except Exception:
            pass
        try:
            pipe.enable_vae_slicing()
        except Exception:
            pass
        pipe = pipe.to(_device)
        _pipeline = pipe
        logger.info("Z-Image-Turbo loaded on device=%s", _device)
        return _pipeline
    except Exception as e:
        logger.exception("Failed to load Z-Image-Turbo: %s", e)
        return None


async def get_pipeline() -> Any:
    """Async access to singleton pipeline; init on first use."""
    global _pipeline
    async with _lock:
        if _pipeline is not None:
            return _pipeline
        loop = asyncio.get_event_loop()
        _pipeline = await loop.run_in_executor(None, _load_pipeline_sync)
        return _pipeline


def _run_inference_sync(
    image_bytes: bytes,
    prompt: str,
    negative_prompt: str,
    strength: float,
    num_steps: int,
    size: int,
    seed: int | None,
) -> bytes:
    """Blocking inference run in executor."""
    import torch
    from PIL import Image
    global _pipeline, _device, _use_autocast
    if _pipeline is None:
        raise RuntimeError("Model not loaded")
    img = Image.open(io.BytesIO(image_bytes)).convert("RGB")
    if img.width != size or img.height != size:
        img = img.resize((size, size), Image.Resampling.LANCZOS)
    gen = torch.Generator(device=_device)
    if seed is not None:
        gen.manual_seed(seed)
    kwargs = dict(
        prompt=prompt,
        image=img,
        strength=strength,
        num_inference_steps=num_steps,
        guidance_scale=0.0,
        generator=gen,
        output_type="pil",
        negative_prompt=negative_prompt or None,
    )
    if _use_autocast and _device.type in ("cuda", "mps"):
        with torch.amp.autocast(device_type=_device.type):
            out = _pipeline(**kwargs)
    else:
        out = _pipeline(**kwargs)
    images = out.images if hasattr(out, "images") else [out[0]]
    if not images:
        raise RuntimeError("No output image")
    buf = io.BytesIO()
    images[0].save(buf, format="PNG")
    return buf.getvalue()


async def run_image_to_image(
    image_bytes: bytes,
    style_key: str,
    custom_prompt: str | None = None,
    strength: float | None = None,
    num_steps: int | None = None,
    size: int | None = None,
    seed: int | None = None,
) -> tuple[bytes, float]:
    """
    Run Z-Image-Turbo img2img. Returns (png_bytes, processing_time_seconds).
    Raises ValueError for invalid input, RuntimeError for model failure.
    """
    pipeline = await get_pipeline()
    if pipeline is None:
        raise RuntimeError("Image model not available. Install torch and diffusers>=0.36 with GPU/CUDA or CPU.")
    settings = get_settings()
    prompt = merge_prompt(style_key, custom_prompt)
    negative_prompt = "blurry, low quality, distorted, watermark, text"
    strength = strength if strength is not None else settings.default_strength
    num_steps = num_steps if num_steps is not None else settings.default_steps
    size = size if size is not None else settings.default_size
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
            size,
            seed,
        ),
    )
    elapsed = time.perf_counter() - start
    return result, elapsed


def is_gpu_available() -> bool:
    """Return True if running on GPU (CUDA or MPS)."""
    try:
        import torch
        if torch.cuda.is_available():
            return True
        if getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
            return True
    except Exception:
        pass
    return False
