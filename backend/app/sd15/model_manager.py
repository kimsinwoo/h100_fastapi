"""
SD 1.5 Img2Img once at startup, warmup, DPM++ 2M Karras.
LoRA: unload before load, weight 1.0, no stacking.
"""
from __future__ import annotations

import logging
import threading
from pathlib import Path
from typing import TYPE_CHECKING

from app.sd15.config import get_settings

if TYPE_CHECKING:
    from diffusers import StableDiffusionImg2ImgPipeline

logger = logging.getLogger(__name__)
_lock = threading.RLock()
_pipeline: StableDiffusionImg2ImgPipeline | None = None
_loaded_lora_path: str | None = None
_model_loaded = False


def _get_dtype():  # noqa: ANN202
    import torch
    return torch.float16 if get_settings().torch_dtype == "float16" else torch.float32


def _device() -> str:
    import torch
    s = get_settings()
    if s.device != "cuda" or not torch.cuda.is_available():
        raise RuntimeError("CUDA only and required")
    return "cuda"


def load_pipeline_at_startup() -> StableDiffusionImg2ImgPipeline:
    global _pipeline, _model_loaded
    with _lock:
        if _pipeline is not None:
            return _pipeline
    from diffusers import DPMSolverMultistepScheduler, StableDiffusionImg2ImgPipeline
    import torch
    s = get_settings()
    device = _device()
    dtype = _get_dtype()
    # 로컬 경로가 있고 존재하면 로컬, 없으면 Hugging Face에서 다운로드
    if s.model_path and s.model_path.exists():
        model_path_or_id = str(s.model_path.resolve())
        logger.info("Loading SD 1.5 Img2Img from local: %s", model_path_or_id)
    else:
        model_path_or_id = s.model_id
        logger.info("Loading SD 1.5 Img2Img from Hugging Face: %s (로컬 캐시 사용)", model_path_or_id)
    pipe = StableDiffusionImg2ImgPipeline.from_pretrained(
        model_path_or_id, torch_dtype=dtype, safety_checker=None,
    )
    pipe.scheduler = DPMSolverMultistepScheduler.from_config(
        pipe.scheduler.config, use_karras_sigmas=True,
    )
    pipe = pipe.to(device)
    if hasattr(pipe, "enable_attention_slicing"):
        pipe.enable_attention_slicing()
    try:
        if hasattr(pipe, "enable_xformers_memory_efficient_attention"):
            pipe.enable_xformers_memory_efficient_attention()
    except Exception as e:
        logger.warning("xformers not available: %s", e)
    with _lock:
        _pipeline = pipe
        _model_loaded = True
    return pipe


def get_pipeline() -> StableDiffusionImg2ImgPipeline:
    with _lock:
        if _pipeline is None:
            raise RuntimeError("SD 1.5 model not loaded")
        return _pipeline


def is_model_loaded() -> bool:
    with _lock:
        return _model_loaded


def warmup() -> None:
    import torch
    from PIL import Image
    pipe = get_pipeline()
    gen = torch.Generator(device=pipe.device).manual_seed(42)
    with torch.no_grad():
        pipe(
            prompt="warmup", image=Image.new("RGB", (512, 512), (0, 0, 0)),
            strength=0.7, num_inference_steps=2, guidance_scale=8.0, generator=gen,
        )
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    logger.info("SD 1.5 warmup done")


def unload_lora(pipe: StableDiffusionImg2ImgPipeline) -> None:
    global _loaded_lora_path
    with _lock:
        if _loaded_lora_path is None:
            return
        try:
            pipe.unload_lora_weights()
        except Exception as e:
            logger.warning("LoRA unload: %s", e)
        _loaded_lora_path = None


def load_lora(pipe: StableDiffusionImg2ImgPipeline, lora_path: str | Path) -> None:
    global _loaded_lora_path
    s = get_settings()
    path = Path(lora_path)
    if not path.exists():
        raise FileNotFoundError(f"LoRA not found: {path}")
    with _lock:
        unload_lora(pipe)
        pipe.load_lora_weights(str(path.resolve()))
        _loaded_lora_path = str(path.resolve())
        try:
            if hasattr(pipe, "set_adapters") and hasattr(pipe, "get_list_adapters"):
                ad = pipe.get_list_adapters()
                if ad:
                    pipe.set_adapters(ad, adapter_weights=[s.lora_weight])
        except Exception:
            pass
    logger.info("LoRA loaded: %s", _loaded_lora_path)


def ensure_lora(pipe: StableDiffusionImg2ImgPipeline, lora_path: str | Path | None) -> None:
    if lora_path is None:
        unload_lora(pipe)
    else:
        load_lora(pipe, lora_path)


def resolve_lora_for_style(style: str) -> Path | None:
    s = get_settings()
    d = s.lora_dir
    if not d.exists():
        return None
    for p in [d / style, d / f"{style}.safetensors", d / style / "pytorch_lora_weights.safetensors"]:
        if p.exists():
            if p.is_dir():
                sf = p / "pytorch_lora_weights.safetensors"
                return sf if sf.exists() else p
            return p
    return None
