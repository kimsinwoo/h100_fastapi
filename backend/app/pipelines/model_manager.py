"""
Lazy-loading, thread-safe model manager. Caches SDXL and Animagine XL pipelines.
"""

from __future__ import annotations

import logging
import threading
from typing import TYPE_CHECKING

from app.core.config import get_settings
from app.models.style_router import ANIMAGINE_XL, SDXL_BASE

if TYPE_CHECKING:
    from diffusers import StableDiffusionXLImg2ImgPipeline, StableDiffusionXLPipeline

logger = logging.getLogger(__name__)

_lock = threading.RLock()
_txt2img: dict[str, StableDiffusionXLPipeline] = {}
_img2img: dict[str, StableDiffusionXLImg2ImgPipeline] = {}


def _device() -> str:
    try:
        import torch
        if torch.cuda.is_available():
            return "cuda"
    except Exception:
        pass
    return "cpu"


def _dtype():
    import torch
    s = get_settings()
    if s.torch_dtype == "bfloat16" and getattr(torch, "bfloat16", None) is not None:
        return torch.bfloat16
    return torch.float16


def _apply_optimizations(pipe: object) -> None:
    s = get_settings()
    if s.enable_vae_slicing and hasattr(pipe, "enable_vae_slicing"):
        pipe.enable_vae_slicing()
    if s.enable_attention_slicing and hasattr(pipe, "enable_attention_slicing"):
        pipe.enable_attention_slicing()
    if s.enable_xformers and hasattr(pipe, "enable_xformers_memory_efficient_attention"):
        try:
            pipe.enable_xformers_memory_efficient_attention()
        except Exception as e:
            logger.warning("xformers not available: %s", e)


def _make_scheduler():
    from diffusers import DPMSolverMultistepScheduler
    return DPMSolverMultistepScheduler.from_pretrained(
        "stabilityai/stable-diffusion-xl-base-1.0",
        subfolder="scheduler",
        use_karras_sigmas=True,
    )


def get_txt2img_pipeline(model_key: str) -> StableDiffusionXLPipeline:
    from diffusers import StableDiffusionXLPipeline

    with _lock:
        if model_key in _txt2img:
            return _txt2img[model_key]

    s = get_settings()
    model_id = s.sdxl_base_id if model_key == SDXL_BASE else s.animagine_xl_id
    device = _device()
    dtype = _dtype()

    logger.info("Loading txt2img pipeline: %s (%s)", model_id, model_key)
    pipe = StableDiffusionXLPipeline.from_pretrained(
        model_id,
        torch_dtype=dtype,
        use_safetensors=True,
    )
    pipe.scheduler = _make_scheduler()
    pipe = pipe.to(device)
    _apply_optimizations(pipe)

    with _lock:
        _txt2img[model_key] = pipe
    return pipe


def get_img2img_pipeline(model_key: str) -> StableDiffusionXLImg2ImgPipeline:
    from diffusers import StableDiffusionXLImg2ImgPipeline

    with _lock:
        if model_key in _img2img:
            return _img2img[model_key]

    txt_pipe = get_txt2img_pipeline(model_key)
    device = _device()
    dtype = _dtype()

    pipe = StableDiffusionXLImg2ImgPipeline(
        vae=txt_pipe.vae,
        text_encoder=txt_pipe.text_encoder,
        text_encoder_2=txt_pipe.text_encoder_2,
        tokenizer=txt_pipe.tokenizer,
        tokenizer_2=txt_pipe.tokenizer_2,
        unet=txt_pipe.unet,
        scheduler=_make_scheduler(),
    )
    pipe = pipe.to(device)
    _apply_optimizations(pipe)

    with _lock:
        _img2img[model_key] = pipe
    return pipe


def is_loaded(model_key: str) -> bool:
    with _lock:
        return model_key in _txt2img
