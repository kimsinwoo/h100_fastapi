"""
Inference: style-only model lookup. No global pipe. No fallback.
Logging: [INFERENCE] style=... model=...
"""
from __future__ import annotations

import random

from app.sdxl_prod.model_registry import get_registry
from app.sdxl_prod.schemas import GenerateRequest, GenerateResponse
from app.sdxl_prod.style_enum import Style
from app.sdxl_prod.utils import encode_bytes_to_base64, image_to_bytes_png


def run_inference(request: GenerateRequest, style_enum: Style) -> GenerateResponse:
    """Run inference using the pipeline for style_enum only. No shared model."""
    registry = get_registry()
    pipe = registry.get(style_enum)

    model_name = getattr(getattr(pipe, "config", None), "_name_or_path", "unknown")
    print(f"[INFERENCE] style={style_enum} model={model_name}", flush=True)

    seed = request.seed if request.seed is not None else random.randint(0, 2**32 - 1)
    import torch
    generator = torch.Generator(device=pipe.device).manual_seed(seed)

    with torch.inference_mode():
        result = pipe(
            prompt=request.prompt,
            negative_prompt=request.negative_prompt or "",
            num_inference_steps=request.steps,
            guidance_scale=request.cfg,
            generator=generator,
            width=request.width,
            height=request.height,
        )

    if not result.images:
        raise RuntimeError("No output image")
    img = result.images[0]
    image_bytes = image_to_bytes_png(img)
    return GenerateResponse(
        image_base64=encode_bytes_to_base64(image_bytes),
        seed=seed,
        style=style_enum.value,
        width=img.width,
        height=img.height,
    )
