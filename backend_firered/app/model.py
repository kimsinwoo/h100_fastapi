"""
Model layer: HunyuanImage-3.0-Instruct singleton and image-to-image inference.
Single backend — no SDXL/Flux/Qwen/FireRed. H100-optimized with FlashInfer MoE.
Forces CUDA + bfloat16; no CPU fallback.
"""

from __future__ import annotations

import logging
import time
from pathlib import Path
from typing import Any

from app.config import get_settings
from app.utils import get_gpu_memory_mb

logger = logging.getLogger(__name__)

_model: Any = None


def _assert_cuda_and_log() -> None:
    """Crash startup if CUDA not available. Log device name."""
    import torch
    assert torch.cuda.is_available(), "CUDA not available — cannot run on H100. Fix environment and restart."
    device_name = torch.cuda.get_device_name(0)
    logger.info("CUDA device: %s", device_name)
    print(f"Device: {device_name}")


def get_model() -> Any:
    """Return the globally loaded model (singleton). Raises RuntimeError if not loaded."""
    if _model is None:
        raise RuntimeError("Model not loaded. Ensure application startup has completed.")
    return _model


def is_loaded() -> bool:
    """Return True if the model singleton is loaded."""
    return _model is not None


def load_model() -> None:
    """
    Load HunyuanImage-3.0-Instruct once (singleton) on CUDA with bfloat16.
    No CPU fallback. TF32 enabled for H100. Call _assert_cuda_and_log() before this at startup.
    """
    global _model
    import torch

    if _model is not None:
        logger.info("Model already loaded, skipping")
        return

    from transformers import AutoModelForCausalLM

    # H100: prefer bfloat16; do not use float32 on GPU
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True

    settings = get_settings()
    model_id = str(settings.model_path())
    if not Path(model_id).exists():
        raise FileNotFoundError(
            f"HunyuanImage model path not found: {model_id}. "
            "Download with: huggingface-cli download tencent/HunyuanImage-3.0-Instruct --local-dir ./HunyuanImage-3-Instruct"
        )

    # Force CUDA + bfloat16 (no torch_dtype="auto" which may yield float32)
    kwargs: dict[str, Any] = {
        "trust_remote_code": True,
        "attn_implementation": "sdpa",
        "torch_dtype": torch.bfloat16,
        "device_map": "auto",
        "moe_impl": "flashinfer",
        "moe_drop_tokens": True,
    }
    logger.info("Loading with torch_dtype=bfloat16, device_map=auto, moe_impl=flashinfer")

    logger.info("Loading model from %s", model_id)
    model = AutoModelForCausalLM.from_pretrained(model_id, **kwargs)
    model.load_tokenizer(model_id)
    _model = model

    # Verify model is on CUDA (no CPU fallback)
    first_param = next(model.parameters())
    assert first_param.is_cuda, "Model loaded on CPU — expected CUDA. Check device_map and GPU visibility."
    device_str = str(first_param.device)
    dtype_str = str(first_param.dtype)
    logger.info("Loaded on %s dtype=%s Device: %s", device_str, dtype_str, torch.cuda.get_device_name(0))
    print(f"Loaded on {device_str}")
    print(f"dtype={dtype_str}")
    print(f"Device: {torch.cuda.get_device_name(0)}")


def generate_image(
    prompt: str,
    image_paths: list[str],
    seed: int = 42,
    image_size: str = "auto",
    use_system_prompt: str = "en_unified",
    bot_task: str = "think_recaption",
    infer_align_image_size: bool = True,
    diff_infer_steps: int = 28,
    verbose: int = 1,
    request_id: str = "",
) -> Any:
    """
    Run image-to-image generation/editing on GPU. Batch size 1.
    Uses inference_mode + autocast(cuda, bfloat16). Crashes if model on CPU.
    """
    import torch

    model = get_model()
    settings = get_settings()

    # Crash if model on CPU (no silent fallback)
    first_param = next(model.parameters())
    assert first_param.is_cuda, "Model on CPU — cannot run inference. Restart with CUDA."

    image_size = image_size or settings.DEFAULT_IMAGE_SIZE
    use_system_prompt = use_system_prompt or settings.DEFAULT_USE_SYSTEM_PROMPT
    bot_task = bot_task or settings.DEFAULT_BOT_TASK
    diff_infer_steps = min(diff_infer_steps, 50)

    # Log runtime device before inference (outside hot path)
    mem_before = get_gpu_memory_mb()
    device_name = torch.cuda.get_device_name(0) if torch.cuda.is_available() else "N/A"
    logger.info(
        "request_id=%s model_device=%s cuda_device=%s memory_allocated_mb=%s requested_steps=%s",
        request_id,
        first_param.device,
        device_name,
        mem_before,
        diff_infer_steps,
    )

    # Optional: scheduler sanity check if model exposes it
    scheduler = getattr(model, "scheduler", None)
    if scheduler is not None and hasattr(scheduler, "timesteps"):
        logger.info("request_id=%s scheduler_timesteps_len=%s", request_id, len(scheduler.timesteps))

    t0 = time.perf_counter()
    with torch.inference_mode(), torch.autocast(device_type="cuda", dtype=torch.bfloat16):
        cot_text, samples = model.generate_image(
            prompt=prompt,
            image=image_paths,
            seed=seed,
            image_size=image_size,
            use_system_prompt=use_system_prompt,
            bot_task=bot_task,
            infer_align_image_size=infer_align_image_size,
            diff_infer_steps=diff_infer_steps,
            verbose=verbose,
        )

    elapsed = time.perf_counter() - t0
    mem_after = get_gpu_memory_mb()

    logger.info(
        "request_id=%s prompt_len=%s num_images=%s resolution=%s steps=%s inference_sec=%.2f gpu_mb_before=%s gpu_mb_after=%s seed=%s",
        request_id,
        len(prompt),
        len(image_paths),
        image_size,
        diff_infer_steps,
        elapsed,
        mem_before,
        mem_after,
        seed,
    )

    if not samples:
        raise RuntimeError("Model returned no images")
    return samples[0]
