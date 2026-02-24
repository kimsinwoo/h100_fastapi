"""
Single GPU worker, asyncio queue. Style-aware inference: STYLE_REGISTRY, overrides, LoRA, scheduler.
"""
from __future__ import annotations

import asyncio
import logging
import random
from typing import NamedTuple

from PIL import Image

from app.sd15.config import get_settings
from app.sd15.model_manager import (
    ensure_lora,
    get_pipeline,
    is_model_loaded,
    resolve_lora_for_style,
    set_scheduler,
    unload_lora,
)
from app.sd15.prompt_engine import (
    build_negative_prompt,
    build_positive_prompt,
    get_effective_params,
    style_enable_upscale,
    style_output_grayscale,
)
from app.sd15.schemas import GenerateRequest, GenerateResponse
from app.sd15.upscale import upscale_image_if_enabled
from app.sd15.utils import (
    decode_base64_to_bytes,
    decode_image_and_validate,
    encode_bytes_to_base64,
    image_to_bytes_png,
    image_to_grayscale,
)

logger = logging.getLogger(__name__)

class _Task(NamedTuple):
    request: GenerateRequest
    future: asyncio.Future[GenerateResponse]

_queue: asyncio.Queue[_Task | None] = asyncio.Queue()
_worker_started = False


def _run_inference(req: GenerateRequest) -> GenerateResponse:
    import torch
    s = get_settings()
    pipe = get_pipeline()
    seed = random.randint(0, 2**32 - 1)
    generator = torch.Generator(device=pipe.device).manual_seed(seed)

    style_key = req.style
    # Merge prompts from STYLE_REGISTRY
    final_prompt = build_positive_prompt(req.prompt, style_key)
    negative = build_negative_prompt(style_key)

    # Style overrides: if client did not send cfg/steps/strength, use style defaults; else respect and clamp
    cfg, steps, strength, sampler_name = get_effective_params(
        style_key,
        req.cfg,
        req.steps,
        req.strength,
        s,
    )

    # Per-style scheduler (e.g. Euler a for pixel_art)
    set_scheduler(pipe, sampler_name)

    # Optional LoRA by style (optional_lora key from registry)
    from app.sd15.style_registry import get_style_config
    config = get_style_config(style_key)
    lora_key = config["optional_lora"]
    lora_path = resolve_lora_for_style(lora_key)
    try:
        ensure_lora(pipe, lora_path)
    except FileNotFoundError as e:
        unload_lora(pipe)
        raise e

    if req.image_base64:
        raw = decode_base64_to_bytes(req.image_base64)
        init_image, _, _ = decode_image_and_validate(raw, max_side=s.max_resolution)
    else:
        init_image = Image.new("RGB", (512, 512), (0, 0, 0))

    try:
        with torch.no_grad():
            out = pipe(
                prompt=final_prompt,
                negative_prompt=negative,
                image=init_image,
                strength=strength,
                num_inference_steps=steps,
                guidance_scale=cfg,
                generator=generator,
            )
    except torch.cuda.OutOfMemoryError:
        unload_lora(pipe)
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        raise RuntimeError("CUDA out of memory")
    if not out.images:
        raise RuntimeError("No output image")

    out_img = out.images[0]
    if style_output_grayscale(style_key):
        out_img = image_to_grayscale(out_img)
    image_bytes = image_to_bytes_png(out_img)

    upscale_requested = style_enable_upscale(style_key, req.upscale)
    image_bytes = upscale_image_if_enabled(image_bytes, upscale_requested=upscale_requested)

    del out, init_image, generator
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    return GenerateResponse(
        image_base64=encode_bytes_to_base64(image_bytes),
        seed=seed,
        style=style_key,
    )


async def _worker_loop(loop: asyncio.AbstractEventLoop) -> None:
    s = get_settings()
    timeout = s.worker_timeout_seconds
    while True:
        task: _Task | None = await _queue.get()
        if task is None:
            break
        req, future = task
        try:
            result = await asyncio.wait_for(
                loop.run_in_executor(None, lambda r=req: _run_inference(r)),
                timeout=timeout,
            )
            if not future.done():
                future.set_result(result)
        except asyncio.CancelledError:
            if not future.done():
                future.cancel()
            raise
        except Exception as e:
            logger.exception("SD15 worker failed: %s", e)
            if not future.done():
                future.set_exception(e)
        finally:
            if is_model_loaded():
                try:
                    unload_lora(get_pipeline())
                except Exception:
                    pass


def start_worker() -> None:
    global _worker_started
    if _worker_started:
        return
    _worker_started = True
    loop = asyncio.get_event_loop()
    asyncio.ensure_future(_worker_loop(loop))
    logger.info("SD 1.5 GPU worker started")


async def enqueue(request: GenerateRequest) -> GenerateResponse:
    s = get_settings()
    if _queue.qsize() >= s.queue_max_size:
        raise RuntimeError("Queue full, try again later")
    loop = asyncio.get_event_loop()
    future: asyncio.Future[GenerateResponse] = loop.create_future()
    await _queue.put(_Task(request=request, future=future))
    return await asyncio.wait_for(future, timeout=s.worker_timeout_seconds + 10.0)
