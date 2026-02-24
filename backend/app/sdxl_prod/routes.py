"""
FastAPI routes for SDXL production: POST /generate, GET /health.
Structured errors, validation, queue submit with timeout.
"""
from __future__ import annotations

import logging
from typing import Any

from fastapi import APIRouter, HTTPException, Request

from app.sdxl_prod.inference import _run_inference
from app.sdxl_prod.lora_manager import preload_lora_paths
from app.sdxl_prod.model_manager import initialize_all_pipelines
from app.sdxl_prod.schemas import GenerateRequest, GenerateResponse
from app.sdxl_prod.worker_queue import WorkerQueue

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/sdxl", tags=["sdxl"])

_worker_queue: WorkerQueue | None = None


def run_sdxl_prod_startup() -> None:
    """Call from app lifespan: preload pipelines, LoRA paths, start worker queue."""
    try:
        initialize_all_pipelines()
        preload_lora_paths(["watercolor", "oil_painting", "sketch"])
        get_worker_queue()
        logger.info("SDXL production startup complete")
    except Exception as e:
        logger.warning("SDXL production startup failed: %s", e)


def get_worker_queue() -> WorkerQueue:
    global _worker_queue
    if _worker_queue is None:
        _worker_queue = WorkerQueue()
        _worker_queue.start()
    return _worker_queue


@router.get("/health")
async def health() -> dict[str, Any]:
    from app.sdxl_prod.model_manager import list_loaded_model_keys
    return {
        "status": "ok",
        "models_loaded": list_loaded_model_keys(),
    }


@router.post("/generate", response_model=GenerateResponse)
async def generate(request: Request, req: GenerateRequest) -> GenerateResponse:
    queue = get_worker_queue()
    try:
        result = await queue.submit(lambda r=req: _run_inference(r))
        return result
    except ValueError as e:
        logger.info("Validation error: %s", e)
        raise HTTPException(status_code=400, detail=str(e))
    except TimeoutError as e:
        logger.warning("Inference timeout: %s", e)
        raise HTTPException(status_code=504, detail="Inference timeout")
    except RuntimeError as e:
        if "OOM" in str(e) or "out of memory" in str(e).lower():
            raise HTTPException(status_code=503, detail="GPU OOM")
        raise HTTPException(status_code=500, detail=str(e))
    except KeyError as e:
        raise HTTPException(status_code=503, detail=f"Model not available: {e}")
    except Exception as e:
        logger.exception("Generate failed: %s", e)
        raise HTTPException(status_code=500, detail="Internal server error")
