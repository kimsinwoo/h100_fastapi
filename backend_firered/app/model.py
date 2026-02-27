"""
Model layer: FireRed-Image-Edit-1.0 singleton and inference.
Factory-friendly for future FireRed-Image-Edit-1.0-Distilled or LoRA.
"""

from __future__ import annotations

import logging
import time
from typing import Any

from PIL import Image

from app.config import get_settings
from app.utils import get_gpu_memory_mb

logger = logging.getLogger(__name__)

# Singleton pipeline. Only mutated at startup.
_pipeline: Any = None
_device: Any = None


def get_pipeline():
    """
    Return the globally loaded pipeline (singleton).
    Raises RuntimeError if the model has not been loaded (startup not completed).
    """
    global _pipeline
    if _pipeline is None:
        raise RuntimeError("Model not loaded. Ensure application startup has completed.")
    return _pipeline


def get_device():
    """Return the device the pipeline is on."""
    global _device
    if _device is None:
        raise RuntimeError("Model not loaded.")
    return _device


def is_loaded() -> bool:
    """Return True if the pipeline singleton is loaded."""
    return _pipeline is not None


def load_pipeline(model_id: str | None = None) -> None:
    """
    Load the FireRed image-edit pipeline once (singleton).
    Uses MODEL_ID from config if model_id is None.
    Future: pass different model_id for Distilled or LoRA variants.
    """
    global _pipeline, _device
    if _pipeline is not None:
        logger.info("Pipeline already loaded, skipping")
        return

    import torch
    from diffusers import QwenImageEditPlusPipeline

    settings = get_settings()
    mid = model_id or settings.MODEL_ID
    dtype = torch.float16 if settings.torch_dtype == "float16" else torch.bfloat16

    # TF32 for Ampere+
    if torch.cuda.is_available():
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True

    logger.info("Loading pipeline: %s (dtype=%s)", mid, dtype)
    pipe = QwenImageEditPlusPipeline.from_pretrained(
        mid,
        torch_dtype=dtype,
        use_safetensors=settings.use_safetensors,
    )
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    pipe = pipe.to(dev)

    if hasattr(pipe, "enable_xformers_memory_efficient_attention"):
        try:
            pipe.enable_xformers_memory_efficient_attention()
            logger.info("xformers memory efficient attention enabled")
        except Exception as e:
            logger.debug("xformers not available: %s", e)

    if hasattr(pipe, "enable_vae_slicing"):
        try:
            pipe.enable_vae_slicing()
            logger.info("VAE slicing enabled")
        except Exception as e:
            logger.debug("enable_vae_slicing: %s", e)

    _pipeline = pipe
    _device = dev
    logger.info("Pipeline loaded on %s", _device)


def run_edit(
    image: Image.Image,
    prompt: str,
    seed: int,
    guidance_scale: float,
    steps: int,
    request_id: str,
) -> Image.Image:
    """
    Run one image edit. Caller must hold the concurrency semaphore.
    Uses inference_mode and autocast(fp16). Batch size is 1.
    """
    import torch

    pipe = get_pipeline()
    device = get_device()
    settings = get_settings()
    dtype = torch.float16 if settings.torch_dtype == "float16" else torch.bfloat16

    generator = torch.Generator(device=device).manual_seed(seed)

    mem_before = get_gpu_memory_mb()
    t0 = time.perf_counter()

    with torch.inference_mode(), torch.autocast(device_type="cuda", dtype=dtype):
        result = pipe(
            image=[image],
            prompt=prompt,
            generator=generator,
            true_cfg_scale=guidance_scale,
            negative_prompt=" ",
            num_inference_steps=steps,
            num_images_per_prompt=1,
        )

    elapsed = time.perf_counter() - t0
    mem_after = get_gpu_memory_mb()

    logger.info(
        "request_id=%s inference_sec=%.2f gpu_mb_before=%s gpu_mb_after=%s seed=%s",
        request_id,
        elapsed,
        mem_before,
        mem_after,
        seed,
    )

    out_images = getattr(result, "images", None)
    if not out_images:
        raise RuntimeError("Pipeline returned no images")
    return out_images[0]
