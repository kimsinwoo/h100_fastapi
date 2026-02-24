"""
SDXL production routes: enum-based style only. Invalid style → 400. No fallback.
"""
from __future__ import annotations

import logging
from typing import Any

from fastapi import APIRouter, HTTPException

from app.sdxl_prod.inference import run_inference
from app.sdxl_prod.model_registry import get_registry, initialize_registry
from app.sdxl_prod.schemas import GenerateRequest, GenerateResponse
from app.sdxl_prod.style_enum import Style, style_from_request_value
from app.sdxl_prod.worker_queue import WorkerQueue

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/sdxl", tags=["sdxl"])

_worker_queue: WorkerQueue | None = None


def get_worker_queue() -> WorkerQueue:
    global _worker_queue
    if _worker_queue is None:
        _worker_queue = WorkerQueue()
        _worker_queue.start()
    return _worker_queue


def run_sdxl_prod_startup() -> None:
    """Load one pipeline per style at startup. No shared model. Call once from lifespan."""
    try:
        initialize_registry()
        get_worker_queue()
        logger.info("SDXL production startup complete (enum-based registry)")
    except Exception as e:
        logger.exception("SDXL production startup failed: %s", e)
        raise


@router.get("/health")
async def health() -> dict[str, Any]:
    registry = get_registry()
    loaded = [s.value for s in Style if s in registry]
    return {"status": "ok", "styles_loaded": loaded}


@router.post("/generate", response_model=GenerateResponse)
async def generate(req: GenerateRequest) -> GenerateResponse:
    try:
        style_enum = style_from_request_value(req.style)
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))

    queue = get_worker_queue()

    try:
        return await queue.submit(lambda: run_inference(req))
    except TimeoutError:
        raise HTTPException(status_code=504, detail="Inference timeout")
    except RuntimeError as e:
        if "OOM" in str(e) or "out of memory" in str(e).lower():
            raise HTTPException(status_code=503, detail="GPU OOM")
        raise HTTPException(status_code=500, detail=str(e))
    except Exception as e:
        logger.exception("Generate failed: %s", e)
        raise HTTPException(status_code=500, detail="Internal server error")
