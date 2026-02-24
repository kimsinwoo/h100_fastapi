"""
ModelManager: preload all base pipelines at startup; retrieval by model_key.
No reload at runtime. Pipelines live on GPU persistently.
"""
from __future__ import annotations

import logging
import os
import threading
from typing import TYPE_CHECKING

from app.sdxl_prod.config import get_settings
from app.sdxl_prod.model_keys import ALL_MODEL_KEYS, DEFAULT_MODEL_IDS

if TYPE_CHECKING:
    from diffusers import StableDiffusionXLImg2ImgPipeline

logger = logging.getLogger(__name__)
_lock = threading.RLock()
_pipelines: dict[str, StableDiffusionXLImg2ImgPipeline] = {}
_pipeline_locks: dict[str, threading.Lock] = {}
_initialized = False


def _get_dtype():
    import torch
    s = get_settings()
    return torch.float16 if s.torch_dtype == "float16" else torch.float32


def _setup_torch_backends() -> None:
    import torch
    if getattr(torch.backends.cuda, "matmul", None) is not None:
        torch.backends.cuda.matmul.allow_tf32 = get_settings().enable_tf32


def _load_single_pipeline(model_key: str) -> StableDiffusionXLImg2ImgPipeline:
    from diffusers import DPMSolverMultistepScheduler, StableDiffusionXLImg2ImgPipeline
    import torch

    model_id = os.environ.get(f"SDXL_PROD_MODEL_ID_{model_key.upper()}", DEFAULT_MODEL_IDS.get(model_key))
    if not model_id:
        raise ValueError(f"No model_id for model_key={model_key}")

    device = get_settings().device
    dtype = _get_dtype()
    s = get_settings()

    logger.info("Loading pipeline %s from %s", model_key, model_id)
    pipe = StableDiffusionXLImg2ImgPipeline.from_pretrained(
        model_id,
        torch_dtype=dtype,
        use_safetensors=True,
        variant="fp16" if dtype == torch.float16 else None,
    )
    pipe.scheduler = DPMSolverMultistepScheduler.from_config(
        pipe.scheduler.config, use_karras_sigmas=True,
    )
    pipe = pipe.to(device)
    if not s.safety_checker and hasattr(pipe, "safety_checker"):
        pipe.safety_checker = None
    if s.enable_attention_slicing and hasattr(pipe, "enable_attention_slicing"):
        pipe.enable_attention_slicing()
    if s.enable_vae_slicing and hasattr(pipe, "enable_vae_slicing"):
        pipe.enable_vae_slicing()
    if s.enable_xformers:
        try:
            if hasattr(pipe, "enable_xformers_memory_efficient_attention"):
                pipe.enable_xformers_memory_efficient_attention()
        except Exception as e:
            logger.warning("xformers not available: %s", e)
    if s.enable_torch_compile:
        try:
            pipe.unet = torch.compile(pipe.unet, mode="reduce-overhead", fullgraph=False)
        except Exception as e:
            logger.warning("torch.compile skipped: %s", e)
    return pipe


def initialize_all_pipelines() -> None:
    """Preload all base pipelines at server startup. Call once from lifespan."""
    global _pipelines, _initialized
    with _lock:
        if _initialized:
            return
    _setup_torch_backends()
    failed: list[str] = []
    for model_key in ALL_MODEL_KEYS:
        try:
            pipe = _load_single_pipeline(model_key)
            with _lock:
                _pipelines[model_key] = pipe
                _pipeline_locks[model_key] = threading.Lock()
        except Exception as e:
            logger.exception("Failed to load pipeline %s: %s", model_key, e)
            failed.append(model_key)
    with _lock:
        _initialized = True
    if failed:
        logger.warning("Some pipelines failed to load: %s", failed)


def get_pipeline(model_key: str) -> StableDiffusionXLImg2ImgPipeline:
    with _lock:
        if model_key not in _pipelines:
            raise KeyError(f"Pipeline not loaded: {model_key}")
        return _pipelines[model_key]


def is_pipeline_loaded(model_key: str) -> bool:
    with _lock:
        return model_key in _pipelines


def list_loaded_model_keys() -> list[str]:
    with _lock:
        return list(_pipelines.keys())


def get_pipeline_lock(model_key: str) -> threading.Lock:
    """Per-pipeline lock for LoRA swap isolation (one inference at a time per pipeline)."""
    with _lock:
        if model_key not in _pipeline_locks:
            _pipeline_locks[model_key] = threading.Lock()
        return _pipeline_locks[model_key]


def set_scheduler(pipe: StableDiffusionXLImg2ImgPipeline, sampler_name: str) -> None:
    """Set pipeline scheduler (euler_a or dpmpp_2m_karras)."""
    from diffusers import DPMSolverMultistepScheduler, EulerAncestralDiscreteScheduler
    from app.sdxl_prod.style_registry import SAMPLER_DPMPP_2M_KARRAS, SAMPLER_EULER_A
    config = pipe.scheduler.config
    if sampler_name == SAMPLER_EULER_A:
        pipe.scheduler = EulerAncestralDiscreteScheduler.from_config(config)
    else:
        pipe.scheduler = DPMSolverMultistepScheduler.from_config(
            config, use_karras_sigmas=True,
        )
