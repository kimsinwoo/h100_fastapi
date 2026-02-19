"""
Z-Image-Turbo image-to-image service. Singleton pipeline, loaded once at startup.
GPU if available with autocast, fallback to CPU. Memory optimized.
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
_use_autocast: bool = True


def _ensure_x_pad_token_dtype(module: Any, device: Any, target_dtype: Any) -> None:
    """파이프라인/transformer 내부 모든 x_pad_token을 target_dtype으로 맞춤 (Half/BFloat16 불일치 방지)."""
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


def _load_pipeline_sync() -> Any:
    """Load Z-Image-Turbo pipeline once. Called at startup."""
    global _pipeline, _device, _use_autocast
    try:
        import os
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
    # MPS 메모리 한도 완화 (기본 상한 42GB 넘어서 OOM 나는 경우 대비)
    if _device.type == "mps" and "PYTORCH_MPS_HIGH_WATERMARK_RATIO" not in os.environ:
        os.environ["PYTORCH_MPS_HIGH_WATERMARK_RATIO"] = "0.0"
        logger.info("MPS: PYTORCH_MPS_HIGH_WATERMARK_RATIO=0.0 set to reduce OOM (override with env if needed)")
    _use_autocast = settings.use_autocast and _device.type in ("cuda", "mps")
    # CUDA: float16 고정. bfloat16 사용 시 diffusers 내부에서 Half/BFloat16 불일치로 Index put 오류 발생 가능 (H100 등).
    dtype = torch.float16
    if _device.type == "cpu":
        dtype = torch.float32
    if _device.type == "mps":
        dtype = torch.float32
        _use_autocast = False
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", category=UserWarning, module="torch.amp.autocast_mode")
        try:
            pipe = ZImageImg2ImgPipeline.from_pretrained(
                settings.model_id,
                torch_dtype=dtype,
                low_cpu_mem_usage=True,
            )
            pipe.set_progress_bar_config(disable=False)
            try:
                pipe.enable_attention_slicing()
            except Exception:
                pass
            try:
                pipe.enable_vae_slicing()
            except Exception:
                pass
            try:
                pipe.enable_vae_tiling()
            except Exception:
                pass
            pipe = pipe.to(_device)
            # Half/BFloat16 불일치 방지: transformer 포함 전체 모듈에서 x_pad_token dtype 통일 (H100 등)
            _ensure_x_pad_token_dtype(pipe, _device, dtype)
            logger.info("Z-Image x_pad_token(s) cast to %s (load)", dtype)
        except Exception as e:
            logger.exception("Failed to load Z-Image-Turbo: %s", e)
            return None
    _pipeline = pipe
    logger.info("Z-Image-Turbo loaded on device=%s (Hugging Face, 로컬 추론)", _device)
    return _pipeline


async def get_pipeline() -> Any:
    """Async access to singleton pipeline; init on first use."""
    global _pipeline
    async with _lock:
        if _pipeline is not None:
            return _pipeline
        loop = asyncio.get_event_loop()
        _pipeline = await loop.run_in_executor(None, _load_pipeline_sync)
        return _pipeline


def _tensor_to_pil_safe(tensor: Any) -> "Image.Image":
    """텐서 → NaN 제거 → [0,1] 스케일 → PIL. VAE 출력은 보통 [-1,1]이므로 먼저 정규화."""
    import numpy as np
    from PIL import Image
    t = tensor.cpu().float()
    if t.dim() == 4:
        t = t[0]
    arr = np.nan_to_num(t.numpy(), nan=0.0, posinf=1.0, neginf=-1.0)
    # diffusion VAE 출력은 [-1,1]. 음수 있으면 [-1,1]→[0,1] 변환
    if arr.size > 0 and np.any(arr < -0.01):
        arr = (arr + 1.0) / 2.0
    arr = np.nan_to_num(arr, nan=0.5, posinf=1.0, neginf=0.0)
    arr = np.clip(arr, 0.0, 1.0)
    arr = (arr * 255.0).round().astype(np.uint8)
    arr = np.transpose(arr, (1, 2, 0))
    return Image.fromarray(arr)


def is_pipeline_loaded() -> bool:
    global _pipeline
    return _pipeline is not None


def get_device_type() -> str:
    """Current device type: 'cuda', 'mps', or 'cpu'."""
    global _device
    if _device is None:
        return "cpu"
    return getattr(_device, "type", "cpu")


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
    logger.info("Image generation started (local, %d steps, size=%d)", num_steps, size)
    img = Image.open(io.BytesIO(image_bytes)).convert("RGB")
    if img.width != size or img.height != size:
        img = img.resize((size, size), Image.Resampling.LANCZOS)
    gen = torch.Generator(device=_device)
    if seed is not None:
        gen.manual_seed(seed)
    # 추론 직전에 x_pad_token dtype 재정렬 (로드 시 bfloat16이어도/서버 코드 불일치여도 Half 오류 방지)
    _ensure_x_pad_token_dtype(
        _pipeline, _device, torch.float16 if _device.type == "cuda" else torch.float32
    )
    kwargs = dict(
        prompt=prompt,
        image=img,
        strength=strength,
        num_inference_steps=num_steps,
        guidance_scale=0.0,
        generator=gen,
        output_type="pt",
        negative_prompt=negative_prompt or None,
    )
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", category=RuntimeWarning)
        if _use_autocast and _device.type in ("cuda", "mps"):
            with torch.amp.autocast(device_type=_device.type):
                out = _pipeline(**kwargs)
        else:
            out = _pipeline(**kwargs)
    # output_type="pt" → out.images is a tensor; do not use "if not images" (Tensor bool error)
    if hasattr(out, "images"):
        raw = out.images
    else:
        raw = out[0] if isinstance(out, (list, tuple)) else out
    if raw is None:
        raise RuntimeError("No output image")
    first_img = raw[0] if isinstance(raw, torch.Tensor) and raw.dim() == 4 else raw
    out_pil = _tensor_to_pil_safe(first_img)
    buf = io.BytesIO()
    out_pil.save(buf, format="PNG")
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
    raw_steps = num_steps if num_steps is not None else settings.default_steps
    # FlowMatchEulerDiscreteScheduler 오류 방지: 9 이상이면 sigmas 인덱스 초과 가능 → 최대 8
    num_steps = min(raw_steps, 8)
    size = size if size is not None else settings.default_size
    # MPS 메모리 절약: 해상도 상한 적용
    if get_device_type() == "mps" and settings.mps_max_size > 0:
        size = min(size, settings.mps_max_size)
        logger.info("MPS: using max size %d to reduce memory", size)
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
