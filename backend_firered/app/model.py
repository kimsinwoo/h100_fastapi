"""
Model layer: HunyuanImage-3.0-Instruct singleton and image-to-image inference.
Single backend — no SDXL/Flux/Qwen/FireRed. H100-optimized with FlashInfer MoE.
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
    Load HunyuanImage-3.0-Instruct once (singleton).
    Uses MODEL_ID path (directory name must not contain dots).
    FlashInfer for MoE acceleration on H100; device_map="auto".
    """
    global _model
    if _model is not None:
        logger.info("Model already loaded, skipping")
        return

    from transformers import AutoModelForCausalLM

    settings = get_settings()
    model_id = str(settings.model_path())
    if not Path(model_id).exists():
        raise FileNotFoundError(
            f"HunyuanImage model path not found: {model_id}. "
            "Download with: huggingface-cli download tencent/HunyuanImage-3.0-Instruct --local-dir ./HunyuanImage-3-Instruct"
        )

    kwargs: dict[str, Any] = {
        "trust_remote_code": True,
        "attn_implementation": "sdpa",
        "torch_dtype": "auto",
        "device_map": "auto",
        "moe_impl": "flashinfer",
        "moe_drop_tokens": True,
    }
    logger.info("Loading with moe_impl=flashinfer for H100 (install flashinfer-python==0.5.0 if missing)")

    logger.info("Loading model from %s", model_id)
    model = AutoModelForCausalLM.from_pretrained(model_id, **kwargs)
    model.load_tokenizer(model_id)
    _model = model
    logger.info("HunyuanImage-3.0-Instruct loaded (device_map=auto)")


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
    Run image-to-image generation/editing. Batch size 1.
    image_paths: list of paths (max 3). No .cpu() or unnecessary transfers in hot path.
    Returns the first generated PIL Image. Logs inference time and GPU memory outside hot path.
    """
    model = get_model()
    settings = get_settings()

    image_size = image_size or settings.DEFAULT_IMAGE_SIZE
    use_system_prompt = use_system_prompt or settings.DEFAULT_USE_SYSTEM_PROMPT
    bot_task = bot_task or settings.DEFAULT_BOT_TASK
    diff_infer_steps = min(diff_infer_steps, 50)

    mem_before = get_gpu_memory_mb()
    t0 = time.perf_counter()

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
