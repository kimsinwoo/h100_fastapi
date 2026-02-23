"""
SDXL image generation and LoRA training API. Fully async. Pydantic v2.
"""

from __future__ import annotations

import logging
import uuid
from pathlib import Path
from typing import Annotated

from fastapi import APIRouter, Body, File, Form, HTTPException, UploadFile

from app.api.schemas import (
    GenerateRequest,
    GenerateResponse,
    HealthResponse,
    TrainLoraRequest,
    TrainLoraResponse,
)
from app.core.config import get_settings
from app.models.style_router import list_styles
from app.pipelines import model_manager
from app.services.generation_service import run_generate
from app.training.runner import train_lora_async
from app.utils.gpu import init_gpu

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api", tags=["sdxl"])


@router.get("/health", response_model=HealthResponse)
async def health() -> HealthResponse:
    try:
        import torch
        gpu = torch.cuda.is_available()
    except Exception:
        gpu = False
    models_loaded = list(model_manager._txt2img.keys()) if model_manager._txt2img else []
    return HealthResponse(
        status="ok",
        gpu_available=gpu,
        models_loaded=models_loaded,
    )


@router.post("/generate", response_model=GenerateResponse)
async def generate(
    prompt: Annotated[str, Form()],
    negative_prompt: Annotated[str, Form()] = "",
    style: Annotated[str, Form()] = "realistic",
    strength: Annotated[float, Form()] = 0.75,
    steps: Annotated[int, Form()] = 30,
    cfg: Annotated[float, Form()] = 7.5,
    seed: Annotated[int | None, Form()] = None,
    width: Annotated[int, Form()] = 1024,
    height: Annotated[int, Form()] = 1024,
    lora_path: Annotated[str | None, Form()] = None,
    lora_scale: Annotated[float, Form()] = 0.85,
    image: Annotated[UploadFile | None, File()] = None,
) -> GenerateResponse:
    s = get_settings()
    if steps > s.max_inference_steps or steps < s.min_inference_steps:
        raise HTTPException(400, f"steps must be {s.min_inference_steps}-{s.max_inference_steps}")
    if width > s.max_resolution or height > s.max_resolution:
        raise HTTPException(400, f"resolution max {s.max_resolution}")

    styles = list_styles()
    if style.lower() not in [x.lower() for x in styles]:
        style = "realistic"

    image_bytes: bytes | None = None
    if image and image.filename:
        raw = await image.read()
        if len(raw) > s.upload_max_bytes:
            raise HTTPException(400, f"File too large. Max {s.upload_max_size_mb}MB")
        if raw:
            image_bytes = raw

    try:
        out_bytes, elapsed = await run_generate(
            prompt=prompt,
            negative_prompt=negative_prompt,
            style=style,
            image_bytes=image_bytes,
            strength=strength,
            steps=steps,
            cfg=cfg,
            seed=seed,
            width=width,
            height=height,
            lora_path=lora_path,
            lora_scale=lora_scale,
        )
    except Exception as e:
        logger.exception("Generate failed: %s", e)
        raise HTTPException(503, str(e)) from e

    name = f"{uuid.uuid4().hex}.png"
    out_path = s.generated_dir
    out_path.mkdir(parents=True, exist_ok=True)
    (out_path / name).write_bytes(out_bytes)
    url = f"/static/{s.generated_dir_name}/{name}"
    return GenerateResponse(image_url=url, processing_time_seconds=round(elapsed, 2))


@router.post("/train-lora", response_model=TrainLoraResponse)
async def train_lora_endpoint(body: TrainLoraRequest = Body(...)) -> TrainLoraResponse:
    dataset_path = Path(body.dataset_path)
    if not dataset_path.exists():
        raise HTTPException(400, f"dataset_path not found: {dataset_path}")
    if not (dataset_path / "captions.txt").exists():
        raise HTTPException(400, "dataset_path must contain captions.txt")
    if not (dataset_path / "images").is_dir():
        raise HTTPException(400, "dataset_path must contain images/")

    try:
        result = await train_lora_async(
            dataset_path=body.dataset_path,
            output_path=body.output_path,
            rank=body.rank,
            learning_rate=body.learning_rate,
            steps=body.steps,
            batch_size=body.batch_size,
            resolution=body.resolution,
        )
    except Exception as e:
        logger.exception("Train LoRA failed: %s", e)
        raise HTTPException(503, str(e)) from e

    status = result.get("status", "failed")
    return TrainLoraResponse(
        status=status,
        output_path=result.get("output_path", body.output_path),
        steps=int(result.get("steps", 0)),
        message=result.get("message"),
        error=result.get("error"),
    )


@router.get("/styles")
async def styles() -> dict[str, str]:
    return {s: s for s in list_styles()}
