"""
Inference: style = model. No global pipe. Registry read-only. No other model access.
Safety log: 사용중인 모델 강제 출력.
"""
from __future__ import annotations

import random
import sys

import torch

from app.sdxl_prod.model_registry import get_registry
from app.sdxl_prod.schemas import GenerateRequest, GenerateResponse
from app.sdxl_prod.style_enum import Style
from app.sdxl_prod.utils import encode_bytes_to_base64, image_to_bytes_png


def run_inference(request: GenerateRequest) -> GenerateResponse:
    """Run inference using the pipeline for request.style only. No shared model, no fallback."""
    style = Style(request.style)
    registry = get_registry()
    pipe = registry.get(style)

    model_name = getattr(getattr(pipe, "config", None), "_name_or_path", "unknown")
    print(f"[INFERENCE] style={style.value} 사용중인 모델={model_name}", file=sys.stderr, flush=True)

    seed = request.seed if request.seed is not None else random.randint(0, 2**32 - 1)
    generator = torch.Generator(device="cuda").manual_seed(seed)

    with torch.inference_mode():
        result = pipe(
            prompt=request.prompt,
            negative_prompt=request.negative_prompt or "",
            num_inference_steps=request.steps,
            guidance_scale=request.cfg,
            width=request.width,
            height=request.height,
            generator=generator,
        )

    if not result.images:
        raise RuntimeError("No output image")
    img = result.images[0]
    image_bytes = image_to_bytes_png(img)
    return GenerateResponse(
        image_base64=encode_bytes_to_base64(image_bytes),
        seed=seed,
        style=style.value,
        width=img.width,
        height=img.height,
    )
