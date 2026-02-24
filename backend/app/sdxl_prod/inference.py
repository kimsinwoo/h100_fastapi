"""
Inference: style에 따라 registry에서만 파이프라인 조회. 전역 pipeline 없음. 재할당 없음.
매 요청 실제 사용 모델 경로 로그 출력.
"""
from __future__ import annotations

import logging
import random
import sys
from typing import NamedTuple

import torch

from PIL import Image

from app.sdxl_prod.model_registry import get_registry
from app.sdxl_prod.schemas import GenerateRequest
from app.sdxl_prod.style_enum import Style
from app.sdxl_prod.utils import encode_bytes_to_base64, image_to_bytes_png

logger = logging.getLogger(__name__)


class InferenceResult(NamedTuple):
    image: Image.Image
    seed: int
    style_value: str


def run_inference(request: GenerateRequest) -> InferenceResult:
    """
    Style Enum으로만 모델 결정. registry.get()으로 해당 스타일 전용 파이프라인만 사용.
    전역 pipeline 없음. inference 시 모델 재할당 없음.
    """
    style_enum = Style(request.style)
    registry = get_registry()
    pipe = registry.get(style_enum)

    model_path = getattr(getattr(pipe, "config", None), "_name_or_path", "unknown")
    logger.info("[INFERENCE] style=%s model=%s", style_enum.value, model_path)
    print(f"[INFERENCE] style={style_enum.value} model={model_path}", file=sys.stderr, flush=True)

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
    return InferenceResult(image=result.images[0], seed=seed, style_value=style_enum.value)


def inference_result_to_response(result: InferenceResult) -> dict[str, str | int]:
    """InferenceResult → API 응답 body (image_base64, seed, style, width, height)."""
    image_bytes = image_to_bytes_png(result.image)
    return {
        "image_base64": encode_bytes_to_base64(image_bytes),
        "seed": result.seed,
        "style": result.style_value,
        "width": result.image.width,
        "height": result.image.height,
    }
