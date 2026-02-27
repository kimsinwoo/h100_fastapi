"""
Model layer: FireRed-Image-Edit-1.0 singleton and inference.
H100/Hopper optimized: TF32, xformers, VAE slicing, channels_last, torch.compile, scheduler, warmup.
"""

from __future__ import annotations

import logging
import time
from typing import Any

from PIL import Image

from app.config import get_settings
from app.utils import get_gpu_memory_mb, normalize_resolution

logger = logging.getLogger(__name__)

# Singleton pipeline. Only mutated at startup.
_pipeline: Any = None
_device: Any = None
_warmup_done: bool = False


def get_pipeline() -> Any:
    """
    Return the globally loaded pipeline (singleton).
    Raises RuntimeError if the model has not been loaded (startup not completed).
    """
    global _pipeline
    if _pipeline is None:
        raise RuntimeError("Model not loaded. Ensure application startup has completed.")
    return _pipeline


def get_device() -> Any:
    """Return the device the pipeline is on."""
    global _device
    if _device is None:
        raise RuntimeError("Model not loaded.")
    return _device


def is_loaded() -> bool:
    """Return True if the pipeline singleton is loaded."""
    return _pipeline is not None


def _is_warmup_done() -> bool:
    return _warmup_done


def load_pipeline(model_id: str | None = None) -> None:
    """
    Load the FireRed image-edit pipeline once (singleton).
    Applies full Hopper optimization stack: TF32, xformers, VAE slicing,
    channels_last, torch.compile(UNet), optional scheduler swap.
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

    # Hopper: TF32 immediately
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

    # xformers memory efficient attention (preferred over attention slicing on H100)
    if hasattr(pipe, "enable_xformers_memory_efficient_attention"):
        try:
            pipe.enable_xformers_memory_efficient_attention()
            logger.info("xformers memory efficient attention enabled")
        except Exception as e:
            logger.debug("xformers not available: %s", e)

    # VAE slicing for memory stability
    if hasattr(pipe, "enable_vae_slicing"):
        try:
            pipe.enable_vae_slicing()
            logger.info("VAE slicing enabled")
        except Exception as e:
            logger.debug("enable_vae_slicing: %s", e)

    # Optional: disable sequential CPU offload on H100 (keep model on GPU)
    if hasattr(pipe, "disable_sequential_cpu_offload"):
        try:
            pipe.disable_sequential_cpu_offload()
        except Exception:
            pass

    # UNet channels_last for Hopper
    if hasattr(pipe, "unet") and pipe.unet is not None:
        try:
            pipe.unet.to(memory_format=torch.channels_last)
            logger.info("UNet converted to channels_last")
        except Exception as e:
            logger.warning("channels_last failed: %s", e)

    # Scheduler: try faster scheduler if compatible (cap steps in run_edit)
    if hasattr(pipe, "scheduler") and pipe.scheduler is not None:
        try:
            from diffusers import DPMSolverMultistepScheduler
            pipe.scheduler = DPMSolverMultistepScheduler.from_config(pipe.scheduler.config)
            logger.info("Scheduler set to DPMSolverMultistepScheduler")
        except Exception as e:
            logger.warning("DPMSolverMultistepScheduler not used, keeping default: %s", e)

    # torch.compile UNet for reduce-overhead (Hopper)
    if hasattr(pipe, "unet") and pipe.unet is not None:
        try:
            pipe.unet = torch.compile(
                pipe.unet,
                mode="reduce-overhead",
                fullgraph=True,
            )
            logger.info("UNet torch.compile(mode=reduce-overhead, fullgraph=True) enabled")
        except Exception as e:
            logger.warning("torch.compile(UNet) failed, using eager: %s", e)

    _pipeline = pipe
    _device = dev
    logger.info("Pipeline loaded on %s", _device)


def run_warmup() -> float:
    """
    Run dummy inference (512 res, 4 steps) to trigger CUDA kernel compile and torch.compile graph.
    Returns warmup duration in seconds. Call once after load_pipeline.
    """
    global _warmup_done
    import torch

    if _warmup_done:
        return 0.0

    pipe = get_pipeline()
    device = get_device()
    settings = get_settings()
    dtype = torch.float16 if settings.torch_dtype == "float16" else torch.bfloat16

    # Dummy 512x512 RGB
    dummy = Image.new("RGB", (512, 512), color=(128, 128, 128))
    generator = torch.Generator(device=device).manual_seed(42)

    t0 = time.perf_counter()
    with torch.inference_mode(), torch.autocast(device_type="cuda", dtype=dtype):
        _ = pipe(
            image=[dummy],
            prompt="warmup",
            generator=generator,
            true_cfg_scale=6.5,
            negative_prompt=" ",
            num_inference_steps=4,
            num_images_per_prompt=1,
        )
    elapsed = time.perf_counter() - t0
    _warmup_done = True
    logger.info("Warmup completed in %.2fs (512x512, 4 steps)", elapsed)
    return elapsed


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
    Applies resolution normalization (max 768), guidance clamp, steps cap.
    No gradient tracking, no CPU sync in hot path, no .item() in hot path.
    GPU memory logging is done after the inference block.
    """
    import torch

    pipe = get_pipeline()
    device = get_device()
    settings = get_settings()
    dtype = torch.float16 if settings.torch_dtype == "float16" else torch.bfloat16

    # Resolution: enforce max 768 (downscale if any side > 1024), keep aspect, Lanczos
    image, original_size, resized_size = normalize_resolution(
        image,
        max_side=settings.MAX_RESOLUTION,
        trigger_threshold=settings.MAX_RESOLUTION_INPUT,
        resample=Image.Resampling.LANCZOS,
    )
    res_w, res_h = resized_size

    # Guidance: if <= 0 and not debug, override to default
    if guidance_scale <= 0 and not settings.DEBUG_MODE:
        guidance_scale = settings.DEFAULT_GUIDANCE
    # Steps cap (production)
    steps = min(steps, settings.PRODUCTION_STEPS_CAP)

    generator = torch.Generator(device=device).manual_seed(seed)

    # GPU memory logged outside inference block to avoid sync in hot path
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
    total_inference_sec = time.perf_counter() - t0

    mem_after = get_gpu_memory_mb()
    if total_inference_sec > 3.0:
        logger.warning(
            "request_id=%s inference_sec=%.2f > 3s resolution=%s steps=%s guidance=%.1f",
            request_id,
            total_inference_sec,
            resized_size,
            steps,
            guidance_scale,
        )
    logger.info(
        "request_id=%s resolution=%s steps=%s guidance=%.1f inference_sec=%.2f gpu_mb_after=%s seed=%s",
        request_id,
        resized_size,
        steps,
        guidance_scale,
        total_inference_sec,
        mem_after,
        seed,
    )

    out_images = getattr(result, "images", None)
    if not out_images:
        raise RuntimeError("Pipeline returned no images")
    return out_images[0]
