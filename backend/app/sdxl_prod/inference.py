"""
Single inference run: style config, model/LoRA, scheduler, params; GPU memory log.
"""
from __future__ import annotations

import logging
import random
from typing import TYPE_CHECKING

from PIL import Image

from app.sdxl_prod.config import get_settings
from app.sdxl_prod.lora_manager import apply_lora, unload_lora
from app.sdxl_prod.model_manager import get_pipeline, get_pipeline_lock, set_scheduler
from app.sdxl_prod.schemas import GenerateRequest, GenerateResponse
from app.sdxl_prod.style_registry import get_style_config
from app.sdxl_prod.utils import (
    build_positive_prompt,
    decode_image_and_validate,
    encode_bytes_to_base64,
    image_to_bytes_png,
    validate_base64_input,
)

if TYPE_CHECKING:
    from diffusers import StableDiffusionXLImg2ImgPipeline

logger = logging.getLogger(__name__)


def _log_gpu_memory() -> None:
    try:
        import torch
        if torch.cuda.is_available():
            allocated = torch.cuda.memory_allocated() / (1024 ** 3)
            reserved = torch.cuda.memory_reserved() / (1024 ** 3)
            logger.info("GPU memory: allocated=%.2f GB reserved=%.2f GB", allocated, reserved)
    except Exception:
        pass


def _effective_params(
    style_key: str,
    user_cfg: float | None,
    user_steps: int | None,
    user_strength: float | None,
) -> tuple[float, int, float]:
    """Resolve cfg, steps, strength; user overrides style defaults. Strength clamped to style range."""
    config = get_style_config(style_key)
    rec = config["recommended_strength_range"]
    s_min, s_max = rec["min"], rec["max"]
    cfg = config["default_cfg"] if user_cfg is None else max(1.0, min(20.0, user_cfg))
    steps = config["default_steps"] if user_steps is None else max(1, min(100, user_steps))
    if user_strength is None:
        strength = (s_min + s_max) / 2.0
    else:
        strength = max(s_min, min(s_max, max(0.0, min(1.0, user_strength))))
    return (cfg, steps, strength)


def _run_inference(req: GenerateRequest) -> GenerateResponse:
    """Blocking inference run. Call from WorkerQueue in executor. Per-pipeline lock for LoRA isolation."""
    import torch

    s = get_settings()
    style_key = req.style
    config = get_style_config(style_key)
    model_key = config["model_key"]
    pipe = get_pipeline(model_key)
    plock = get_pipeline_lock(model_key)

    seed = random.randint(0, 2**32 - 1)
    generator = torch.Generator(device=pipe.device).manual_seed(seed)

    positive = build_positive_prompt(config["positive_prompt_template"], req.prompt)
    negative = config["negative_prompt_template"]
    cfg, steps, strength = _effective_params(style_key, req.cfg, req.steps, req.strength)

    set_scheduler(pipe, config["default_sampler"])

    with plock:
        lora_key = config["optional_lora_key"]
        if lora_key:
            try:
                apply_lora(pipe, lora_key)
            except FileNotFoundError:
                logger.warning("LoRA not found for style %s: %s", style_key, lora_key)
        out_w, out_h = 768, 768
        try:
            if req.image_base64:
                raw = validate_base64_input(req.image_base64)
                init_image, w, h = decode_image_and_validate(raw, max_side=s.max_resolution)
                out_w, out_h = w, h
            else:
                res = req.width or req.height or s.default_resolution
                res = max(256, min(1024, res))
                w, h = res, res
                init_image = Image.new("RGB", (w, h), (0, 0, 0))
                out_w, out_h = w, h

            with torch.inference_mode():
                out = pipe(
                    prompt=positive,
                    negative_prompt=negative,
                    image=init_image,
                    strength=strength,
                    num_inference_steps=steps,
                    guidance_scale=cfg,
                    generator=generator,
                )
            if not out.images:
                raise RuntimeError("No output image")
            result_img = out.images[0]
            image_bytes = image_to_bytes_png(result_img)
            out_w, out_h = result_img.width, result_img.height
        finally:
            if lora_key:
                unload_lora(pipe, lora_key)

    _log_gpu_memory()
    return GenerateResponse(
        image_base64=encode_bytes_to_base64(image_bytes),
        seed=seed,
        style=style_key,
        width=out_w,
        height=out_h,
    )
