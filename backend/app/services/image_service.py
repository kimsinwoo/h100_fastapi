"""
Z-Image-Turbo img2img service. Singleton pipeline, float16 + VAE float32, PIL output.
"""

from __future__ import annotations

import asyncio
import io
import logging
import time
import warnings
from typing import Any, Callable

from app.core.config import get_settings
from app.models.style_presets import merge_prompt

logger = logging.getLogger(__name__)

_pipeline: Any = None
_lock = asyncio.Lock()
_device: Any = None

DEFAULT_STRENGTH = 0.65
DEFAULT_GUIDANCE_SCALE = 7.5
DEFAULT_NUM_INFERENCE_STEPS = 35
MODEL_RESOLUTION = 512

PIXEL_POSITIVE_ADD = "strictly 2D, flat shading, hard pixel edges, no gradients"
PIXEL_NEGATIVE_ADD = "3D, voxel, lego, render, ray tracing, depth of field"


def _ensure_x_pad_token_dtype(module: Any, device: Any, target_dtype: Any) -> None:
    import torch
    if module is None:
        return
    if hasattr(module, "x_pad_token") and isinstance(getattr(module, "x_pad_token"), torch.Tensor):
        token = getattr(module, "x_pad_token")
        if token.dtype != target_dtype:
            setattr(module, "x_pad_token", token.to(target_dtype).to(device))
    for _, child in getattr(module, "named_children", lambda: [])():
        _ensure_x_pad_token_dtype(child, device, target_dtype)


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


def _get_model_resolution(pipe: Any) -> int:
    try:
        if hasattr(pipe, "transformer") and getattr(pipe.transformer, "config", None) is not None:
            sample_size = getattr(pipe.transformer.config, "sample_size", None)
            if sample_size is not None:
                return int(sample_size) * 8
        if hasattr(pipe, "unet") and getattr(pipe.unet, "config", None) is not None:
            sample_size = getattr(pipe.unet.config, "sample_size", None)
            if sample_size is not None:
                return int(sample_size) * 8
    except Exception:
        pass
    return MODEL_RESOLUTION


def _load_pipeline_sync() -> Any:
    global _pipeline, _device
    try:
        import os
        import torch
    except ImportError as e:
        logger.error("torch not installed: %s", e)
        return None
    try:
        from diffusers import ZImageImg2ImgPipeline
    except ImportError as e:
        logger.error("diffusers import failed: %s", e)
        return None
    settings = get_settings()
    _device = _resolve_device()
    if _device.type == "mps" and "PYTORCH_MPS_HIGH_WATERMARK_RATIO" not in os.environ:
        os.environ["PYTORCH_MPS_HIGH_WATERMARK_RATIO"] = "0.0"
    dtype = torch.float16 if _device.type == "cuda" else torch.float32
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", category=UserWarning, module="torch.amp.autocast_mode")
        try:
            pipe = ZImageImg2ImgPipeline.from_pretrained(
                settings.model_id,
                torch_dtype=dtype,
                low_cpu_mem_usage=True,
            )
            pipe.set_progress_bar_config(disable=False)
            for method_name in ("enable_attention_slicing", "enable_vae_slicing", "enable_vae_tiling"):
                method = getattr(pipe, method_name, None)
                if callable(method):
                    try:
                        method()
                    except Exception:
                        pass
            pipe = pipe.to(_device)
            if _device.type == "cuda":
                pipe.vae.to(torch.float32)
            _ensure_x_pad_token_dtype(pipe, _device, dtype)
        except Exception as e:
            logger.exception("Failed to load Z-Image-Turbo: %s", e)
            return None
    _pipeline = pipe
    logger.info("Z-Image-Turbo loaded on device=%s", _device)
    return _pipeline


async def get_pipeline() -> Any:
    global _pipeline
    async with _lock:
        if _pipeline is not None:
            return _pipeline
        loop = asyncio.get_event_loop()
        _pipeline = await loop.run_in_executor(None, _load_pipeline_sync)
        return _pipeline


def is_pipeline_loaded() -> bool:
    global _pipeline
    return _pipeline is not None


def get_device_type() -> str:
    global _device
    if _device is None:
        return "cpu"
    return getattr(_device, "type", "cpu")


def _make_nan_callback(device: Any) -> Callable[..., dict]:
    import torch
    def _callback(pipe: Any, step: int, timestep: Any, callback_kwargs: dict) -> dict:
        latents = callback_kwargs.get("latents")
        if latents is not None:
            if torch.isnan(latents).any().item():
                raise ValueError("NaN detected in latents before decode")
        return callback_kwargs
    return _callback


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
        raise RuntimeError("Model not loaded")
    logger.info("Image generation started (local, %d steps, size=%d)", num_steps, size)
    img = Image.open(io.BytesIO(image_bytes)).convert("RGB")
    if img.width != size or img.height != size:
        img = img.resize((size, size), Image.Resampling.LANCZOS)
    generator = torch.Generator(device=_device)
    if seed is not None:
        generator.manual_seed(seed)
    _ensure_x_pad_token_dtype(
        _pipeline, _device, torch.float16 if _device.type == "cuda" else torch.float32
    )
    nan_callback = _make_nan_callback(_device)
    call_kw: dict = {
        "prompt": prompt,
        "image": img,
        "strength": strength,
        "num_inference_steps": num_steps,
        "guidance_scale": guidance_scale,
        "generator": generator,
        "output_type": "pil",
        "negative_prompt": negative_prompt or None,
    }
    with torch.inference_mode():
        try:
            out = _pipeline(
                **call_kw,
                callback_on_step_end=nan_callback,
                callback_on_step_end_tensor_inputs=["latents"],
            )
        except TypeError:
            out = _pipeline(**call_kw)
    if not getattr(out, "images", None) or len(out.images) == 0:
        raise RuntimeError("No output image from pipeline")
    pil_image = out.images[0]
    buf = io.BytesIO()
    pil_image.save(buf, format="PNG")
    logger.info("Image generation done (output %d bytes)", len(buf.getvalue()))
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
    pipeline = await get_pipeline()
    if pipeline is None:
        raise RuntimeError("Image model not available. Install torch and diffusers with GPU/CUDA or CPU.")
    settings = get_settings()
    prompt = merge_prompt(style_key, custom_prompt)
    negative_prompt = "blurry, low quality, distorted, watermark, text"
    if style_key.strip().lower().startswith("pixel") or "pixel" in style_key.strip().lower():
        prompt = f"{prompt}, {PIXEL_POSITIVE_ADD}"
        negative_prompt = f"{negative_prompt}, {PIXEL_NEGATIVE_ADD}"
    strength = strength if strength is not None else DEFAULT_STRENGTH
    strength = max(0.0, min(1.0, strength))
    num_steps = num_steps if num_steps is not None else DEFAULT_NUM_INFERENCE_STEPS
    num_steps = max(1, min(50, num_steps))
    guidance_scale = DEFAULT_GUIDANCE_SCALE
    resolution = _get_model_resolution(pipeline) if size is None else size
    if get_device_type() == "mps" and getattr(settings, "mps_max_size", 0) > 0:
        resolution = min(resolution, settings.mps_max_size)
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


def is_gpu_available() -> bool:
    try:
        import torch
        if torch.cuda.is_available():
            return True
        if getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
            return True
    except Exception:
        pass
    return False
