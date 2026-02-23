"""
Unified txt2img + img2img interface. Style-based model selection.
"""

from __future__ import annotations

import io
import logging
from pathlib import Path
from typing import Optional

from PIL import Image

from app.core.config import get_settings
from app.models.style_router import get_model_key_for_style
from app.pipelines.model_manager import get_img2img_pipeline, get_txt2img_pipeline

logger = logging.getLogger(__name__)


def _dtype():
    import torch
    s = get_settings()
    if s.torch_dtype == "bfloat16" and getattr(torch, "bfloat16", None) is not None:
        return torch.bfloat16
    return torch.float16


def _clamp_resolution(width: int, height: int) -> tuple[int, int]:
    s = get_settings()
    w = max(s.min_resolution, min(s.max_resolution, width))
    h = max(s.min_resolution, min(s.max_resolution, height))
    w = (w // 8) * 8
    h = (h // 8) * 8
    return w, h


def generate_image(
    prompt: str,
    negative_prompt: str,
    style: str,
    image: Optional[Image.Image],
    strength: float,
    steps: int,
    cfg: float,
    seed: Optional[int],
    width: int = 1024,
    height: int = 1024,
    lora_path: Optional[str | Path] = None,
    lora_scale: float = 0.85,
) -> bytes:
    s = get_settings()
    steps = max(s.min_inference_steps, min(s.max_inference_steps, steps))
    cfg = max(1.0, min(20.0, cfg))
    strength = max(0.0, min(1.0, strength))
    width, height = _clamp_resolution(width, height)

    model_key = get_model_key_for_style(style)
    device = "cuda"
    dtype = _dtype()

    import torch
    from app.lora.manager import load_lora, unload_lora

    generator: Optional[torch.Generator] = None
    if seed is not None:
        generator = torch.Generator(device=device).manual_seed(seed)

    if image is None:
        pipe = get_txt2img_pipeline(model_key)
    else:
        pipe = get_img2img_pipeline(model_key)

    try:
        if lora_path:
            load_lora(pipe, lora_path, scale=lora_scale)
        if image is None:
            with torch.inference_mode():
                out = pipe(
                    prompt=prompt,
                    negative_prompt=negative_prompt or None,
                    num_inference_steps=steps,
                    guidance_scale=cfg,
                    width=width,
                    height=height,
                    generator=generator,
                )
        else:
            if image.mode != "RGB":
                image = image.convert("RGB")
            image = image.resize((width, height), Image.Resampling.LANCZOS)
            with torch.inference_mode():
                out = pipe(
                    prompt=prompt,
                    negative_prompt=negative_prompt or None,
                    image=image,
                    strength=strength,
                    num_inference_steps=steps,
                    guidance_scale=cfg,
                    generator=generator,
                )
    finally:
        if lora_path:
            unload_lora(pipe)

    pil = out.images[0]
    buf = io.BytesIO()
    pil.save(buf, format="PNG")
    return buf.getvalue()
