"""
Reels Dance LoRA API routes.

Endpoints:
  GET  /reels-dance/registry              — list available LoRA models
  GET  /reels-dance/categories            — list dance categories
  POST /reels-dance/train                 — submit training job
  GET  /reels-dance/train/{job_id}        — training job status
  POST /reels-dance/generate              — generate dance video (async job)
  GET  /reels-dance/generate/{job_id}     — generation job status

All routes are mounted under the main router with prefix /api.
"""

from __future__ import annotations

import asyncio
import logging
import time
import uuid
from pathlib import Path
from typing import Dict

from fastapi import APIRouter, BackgroundTasks, File, Form, HTTPException, UploadFile
from fastapi.responses import JSONResponse

from app.core.config import get_settings
from app.schemas.reels_schema import (
    LoRAEntryResponse,
    ReelsGenerateJobResponse,
    ReelsGenerateStatusResponse,
    RegistryInfoResponse,
    TrainingJobResponse,
    TrainingRequest,
    TrainingStatusResponse,
)
from app.services.reels_lora.lora_registry import BUILTIN_CATEGORIES, CATEGORY_DISPLAY_NAMES, get_registry
from app.services.reels_lora.inference_extension import ReelsDanceGenerator

logger = logging.getLogger(__name__)
router = APIRouter(prefix="/reels-dance", tags=["reels-dance"])

# ──────────────────────────────────────────────────────────────────────────────
# In-memory job stores
# ──────────────────────────────────────────────────────────────────────────────

_train_jobs: Dict[str, dict] = {}
_gen_jobs: Dict[str, dict] = {}


# ──────────────────────────────────────────────────────────────────────────────
# Registry endpoints
# ──────────────────────────────────────────────────────────────────────────────

@router.get("/registry", response_model=RegistryInfoResponse, summary="List trained LoRA models")
async def get_registry_info():
    """Return all registered LoRA dance models."""
    registry = get_registry()
    registry.scan()  # refresh
    info = registry.info()
    entries = [LoRAEntryResponse(**e) for e in info.entries]
    return RegistryInfoResponse(
        total=info.total,
        categories=info.categories,
        loaded=info.loaded,
        entries=entries,
    )


@router.get("/categories", summary="List available dance categories")
async def list_categories():
    """Return built-in and trained dance categories."""
    registry = get_registry()
    trained = registry.list_categories()
    categories = [
        {
            "id": cat,
            "display_name": CATEGORY_DISPLAY_NAMES.get(cat, cat.replace("_", " ").title()),
            "has_lora": cat in trained,
        }
        for cat in BUILTIN_CATEGORIES
    ]
    # Also include any custom trained categories not in built-in list
    for cat in trained:
        if cat not in BUILTIN_CATEGORIES:
            categories.append({
                "id": cat,
                "display_name": CATEGORY_DISPLAY_NAMES.get(cat, cat.replace("_", " ").title()),
                "has_lora": True,
            })
    return {"categories": categories}


# ──────────────────────────────────────────────────────────────────────────────
# Training endpoints
# ──────────────────────────────────────────────────────────────────────────────

@router.post("/train", response_model=TrainingJobResponse, summary="Submit LoRA training job")
async def submit_training_job(
    request: TrainingRequest,
    background_tasks: BackgroundTasks,
):
    """
    Queue a LoRA training job for a dance category.

    The training runs in the background. Poll `/reels-dance/train/{job_id}` for status.

    ⚠️ Requires dance videos in `video_dir/{category}/` directory.
    """
    settings = get_settings()
    job_id = str(uuid.uuid4())

    # Resolve video_dir relative to backend if not absolute
    video_dir = Path(request.video_dir)
    if not video_dir.is_absolute():
        video_dir = settings.backend_dir / video_dir

    if not video_dir.exists():
        raise HTTPException(
            status_code=400,
            detail=f"video_dir not found: {video_dir}",
        )

    category_dir = video_dir / request.category
    if not category_dir.exists():
        raise HTTPException(
            status_code=400,
            detail=f"Category directory not found: {category_dir}",
        )

    _train_jobs[job_id] = {
        "job_id": job_id,
        "category": request.category,
        "status": "queued",
        "step": 0,
        "total_steps": request.num_train_steps,
        "loss": None,
        "output_path": None,
        "error": None,
        "elapsed_seconds": None,
        "started_at": time.time(),
    }

    background_tasks.add_task(_run_training_job, job_id, request, video_dir)

    logger.info("[API] Training job queued: %s | category=%s", job_id, request.category)
    return TrainingJobResponse(
        job_id=job_id,
        category=request.category,
        status="queued",
        message=f"Training queued. Poll /reels-dance/train/{job_id} for progress.",
    )


@router.get("/train/{job_id}", response_model=TrainingStatusResponse, summary="Training job status")
async def get_training_status(job_id: str):
    if job_id not in _train_jobs:
        raise HTTPException(status_code=404, detail="Training job not found")
    return TrainingStatusResponse(**_train_jobs[job_id])


# ──────────────────────────────────────────────────────────────────────────────
# Generation endpoints
# ──────────────────────────────────────────────────────────────────────────────

@router.post("/generate", response_model=ReelsGenerateJobResponse, summary="Generate dance video")
async def generate_reels_dance(
    background_tasks: BackgroundTasks,
    image: UploadFile = File(..., description="Character image (JPEG/PNG)"),
    category: str = Form(..., description="Dance category, e.g. tiktok_shuffle"),
    character: str = Form(default="dog", description="Character type: dog, cat, etc."),
    mode: str = Form(
        default="ltx2",
        description="Generation mode: 'animatediff' (requires trained LoRA) or 'ltx2'",
    ),
    seed: int = Form(default=None),
    num_frames: int = Form(default=16, ge=8, le=49),
):
    """
    Generate a dance video with optional LoRA motion transfer.

    **mode=animatediff**: Uses trained AnimateDiff LoRA for the category.
    Falls back to LTX-2 if no LoRA weights are available.

    **mode=ltx2**: Always uses existing LTX-2 pipeline with enhanced prompt.

    Returns a job_id for polling.
    """
    settings = get_settings()

    # Validate file type
    if image.content_type not in ("image/jpeg", "image/jpg", "image/png", "image/webp"):
        raise HTTPException(status_code=400, detail="Only JPEG/PNG/WebP images accepted")

    image_bytes = await image.read()
    if len(image_bytes) > settings.upload_max_bytes:
        raise HTTPException(
            status_code=413,
            detail=f"Image too large (max {settings.upload_max_size_mb} MB)",
        )

    if category not in BUILTIN_CATEGORIES:
        registry = get_registry()
        if category not in registry.list_categories():
            raise HTTPException(
                status_code=400,
                detail=f"Unknown category '{category}'. "
                       f"Available: {BUILTIN_CATEGORIES + registry.list_categories()}",
            )

    job_id = str(uuid.uuid4())
    _gen_jobs[job_id] = {
        "job_id": job_id,
        "status": "processing",
        "category": category,
        "character": character,
        "mode": mode,
        "video_url": None,
        "processing_time": None,
        "error": None,
    }

    background_tasks.add_task(
        _run_generation_job,
        job_id, image_bytes, category, character, mode, seed, num_frames
    )

    logger.info(
        "[API] Generation job started: %s | category=%s | mode=%s",
        job_id, category, mode,
    )
    return ReelsGenerateJobResponse(
        job_id=job_id,
        category=category,
        status="processing",
    )


@router.get(
    "/generate/{job_id}",
    response_model=ReelsGenerateStatusResponse,
    summary="Generation job status",
)
async def get_generation_status(job_id: str):
    if job_id not in _gen_jobs:
        raise HTTPException(status_code=404, detail="Generation job not found")
    return ReelsGenerateStatusResponse(**_gen_jobs[job_id])


# ──────────────────────────────────────────────────────────────────────────────
# Background tasks
# ──────────────────────────────────────────────────────────────────────────────

async def _run_training_job(job_id: str, request: TrainingRequest, video_dir: Path) -> None:
    job = _train_jobs[job_id]
    job["status"] = "running"
    t0 = time.time()

    try:
        from app.services.reels_lora.lora_trainer import LoRATrainingConfig, ReelsLoRATrainer

        cfg = LoRATrainingConfig(
            category=request.category,
            video_dir=str(video_dir),
            num_train_steps=request.num_train_steps,
            lora_rank=request.lora_rank,
            lora_alpha=request.lora_alpha,
            learning_rate=request.learning_rate,
            batch_size=request.batch_size,
            clip_frames=request.clip_frames,
            image_size=request.image_size,
            gradient_accumulation_steps=request.gradient_accumulation_steps,
            save_every=request.save_every,
            enable_torch_compile=request.enable_torch_compile,
        )

        loop = asyncio.get_event_loop()
        output_path: Path = await loop.run_in_executor(
            None, _run_trainer_sync, cfg
        )

        # Register trained weights
        registry = get_registry()
        registry.register(request.category, output_path)

        job.update({
            "status": "completed",
            "output_path": str(output_path),
            "elapsed_seconds": time.time() - t0,
        })
        logger.info("[Training] Job %s completed → %s", job_id, output_path)

    except Exception as e:
        logger.exception("[Training] Job %s failed: %s", job_id, e)
        job.update({"status": "failed", "error": str(e), "elapsed_seconds": time.time() - t0})


def _run_trainer_sync(cfg) -> Path:
    from app.services.reels_lora.lora_trainer import ReelsLoRATrainer
    trainer = ReelsLoRATrainer(cfg)
    return trainer.train()


async def _run_generation_job(
    job_id: str,
    image_bytes: bytes,
    category: str,
    character: str,
    mode: str,
    seed,
    num_frames: int,
) -> None:
    job = _gen_jobs[job_id]
    settings = get_settings()
    t0 = time.time()

    try:
        generator = ReelsDanceGenerator()
        video_bytes, elapsed = await generator.generate(
            image_bytes=image_bytes,
            category=category,
            character=character,
            mode=mode,
            seed=seed,
            num_frames=num_frames,
        )

        # Save to static/generated
        filename = f"reels_{job_id}.mp4"
        out_path = settings.generated_dir / filename
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_bytes(video_bytes)

        job.update({
            "status": "completed",
            "video_url": f"/static/generated/{filename}",
            "processing_time": elapsed,
        })
        logger.info("[Generation] Job %s completed in %.1fs", job_id, elapsed)

    except Exception as e:
        logger.exception("[Generation] Job %s failed: %s", job_id, e)
        job.update({"status": "failed", "error": str(e), "processing_time": time.time() - t0})
