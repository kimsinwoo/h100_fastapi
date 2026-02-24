"""
SD 1.5 API: POST /sd15/generate, GET /sd15/health.
Startup: run_sd15_startup() loads model (if path exists) and starts worker.
"""
from __future__ import annotations

import logging
from typing import Any

from fastapi import APIRouter, HTTPException

from app.sd15.config import get_settings
from app.sd15.model_manager import is_model_loaded, load_pipeline_at_startup, warmup
from app.sd15.schemas import GenerateRequest, GenerateResponse
from app.sd15.worker import enqueue, start_worker

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/sd15", tags=["sd15"])


def run_sd15_startup() -> None:
    """Call from main lifespan: load SD 1.5 from Hugging Face or local path, warmup, start worker."""
    try:
        load_pipeline_at_startup()
        warmup()
        start_worker()
        logger.info("SD 1.5 startup done")
    except Exception as e:
        logger.warning("SD15 startup failed (sd15 endpoints disabled): %s", e)


@router.get("/health")
async def sd15_health() -> dict[str, Any]:
    return {
        "status": "ok",
        "model_loaded": is_model_loaded(),
    }


@router.post("/generate", response_model=GenerateResponse)
async def sd15_generate(req: GenerateRequest) -> GenerateResponse:
    if not is_model_loaded():
        raise HTTPException(status_code=503, detail="SD 1.5 model not loaded")
    try:
        return await enqueue(req)
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except RuntimeError as e:
        if "Queue full" in str(e) or "out of memory" in str(e).lower():
            raise HTTPException(status_code=503, detail=str(e))
        raise HTTPException(status_code=500, detail=str(e))
