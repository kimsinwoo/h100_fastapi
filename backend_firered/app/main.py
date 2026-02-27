"""
FireRed-Image-Edit-1.0 FastAPI application.
Single endpoint: POST /edit. Model loaded once at startup (singleton).
"""

from __future__ import annotations

import asyncio
import io
import logging
import random
import time
from contextlib import asynccontextmanager
from typing import Annotated

from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, Response

from app.config import get_settings
from app.model import is_loaded, load_pipeline, run_edit, run_warmup
from app.schemas import error_payload
from app.utils import generate_request_id, get_gpu_memory_mb, get_gpu_info, load_image_rgb

# Structured logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger(__name__)

# Concurrency limit: only this many GPU jobs at a time
_gpu_semaphore: asyncio.Semaphore | None = None


def _get_semaphore() -> asyncio.Semaphore:
    if _gpu_semaphore is None:
        raise RuntimeError("Semaphore not initialized")
    return _gpu_semaphore


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Startup: load model once, run warmup (dummy 512, 4 steps). Shutdown: log."""
    global _gpu_semaphore
    settings = get_settings()
    _gpu_semaphore = asyncio.Semaphore(settings.MAX_CONCURRENT_JOBS)
    logger.info("MAX_CONCURRENT_JOBS=%s TIMEOUT_SECONDS=%s", settings.MAX_CONCURRENT_JOBS, settings.TIMEOUT_SECONDS)
    # Load model in thread so event loop is not blocked
    await asyncio.to_thread(load_pipeline)
    logger.info("Model loaded: %s", is_loaded())
    # Warmup: dummy inference to force CUDA kernel compile and torch.compile graph capture
    warmup_sec = await asyncio.to_thread(run_warmup)
    logger.info(
        "Startup benchmark: warmup=%.2fs | default_steps=%s default_guidance=%.1f max_resolution=%s steps_cap=%s",
        warmup_sec,
        settings.DEFAULT_STEPS,
        settings.DEFAULT_GUIDANCE,
        settings.MAX_RESOLUTION,
        settings.PRODUCTION_STEPS_CAP,
    )
    yield
    logger.info("Shutting down")


app = FastAPI(
    title="FireRed-Image-Edit API",
    version="1.0.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.post("/edit")
async def edit(
    image: Annotated[UploadFile, File(description="Image file to edit")],
    prompt: Annotated[str, Form(description="Editing instruction")],
    seed: Annotated[int | None, Form()] = None,
    guidance_scale: Annotated[float | None, Form()] = None,
    steps: Annotated[int | None, Form()] = None,
) -> Response:
    """
    Edit the uploaded image according to the text prompt.
    Returns PNG image stream (no base64).
    """
    request_id = generate_request_id()
    start_time = time.perf_counter()

    # Validate content type
    if not image.content_type or not image.content_type.startswith("image/"):
        return JSONResponse(
            status_code=400,
            content=error_payload("invalid_file", "File must be an image", request_id),
        )

    try:
        pil_image = await load_image_rgb(image)
    except ValueError as e:
        return JSONResponse(
            status_code=400,
            content=error_payload("invalid_image", str(e), request_id),
        )

    settings = get_settings()
    seed_val = seed if seed is not None else random.randint(0, 2**31 - 1)
    guidance = guidance_scale if guidance_scale is not None else settings.DEFAULT_GUIDANCE
    # Force proper guidance: if <= 0 and not debug, override to default
    if guidance <= 0 and not settings.DEBUG_MODE:
        guidance = settings.DEFAULT_GUIDANCE
    steps_val = steps if steps is not None else settings.DEFAULT_STEPS

    sem = _get_semaphore()
    try:
        await asyncio.wait_for(
            sem.acquire(),
            timeout=float(settings.TIMEOUT_SECONDS),
        )
    except asyncio.TimeoutError:
        return JSONResponse(
            status_code=504,
            content=error_payload("timeout", "Request timed out waiting for GPU", request_id),
        )

    try:
        try:
            # Run inference in thread pool to avoid blocking; with overall timeout
            result_pil = await asyncio.wait_for(
                asyncio.to_thread(
                    run_edit,
                    pil_image,
                    prompt.strip(),
                    seed_val,
                    guidance,
                    steps_val,
                    request_id,
                ),
                timeout=float(settings.TIMEOUT_SECONDS),
            )
        except asyncio.TimeoutError:
            return JSONResponse(
                status_code=504,
                content=error_payload("timeout", "Inference timed out", request_id),
            )
        except RuntimeError as e:
            err_msg = str(e)
            if "out of memory" in err_msg.lower() or "oom" in err_msg.lower():
                return JSONResponse(
                    status_code=503,
                    content=error_payload("gpu_oom", f"GPU out of memory: {err_msg}", request_id),
                )
            return JSONResponse(
                status_code=503,
                content=error_payload("inference_error", err_msg, request_id),
            )

        # Encode PNG and return binary
        buf = io.BytesIO()
        result_pil.save(buf, format="PNG")
        png_bytes = buf.getvalue()

        total_duration = time.perf_counter() - start_time
        logger.info(
            "request_id=%s total_sec=%.2f seed=%s",
            request_id,
            total_duration,
            seed_val,
        )

        return Response(
            content=png_bytes,
            media_type="image/png",
            headers={"X-Request-Id": request_id},
        )
    finally:
        sem.release()


@app.get("/health")
async def health() -> dict[str, str | bool | float | None]:
    """Health check: status, model loaded, optional GPU memory."""
    gpu_mb = get_gpu_memory_mb()
    return {
        "status": "ok",
        "model_loaded": is_loaded(),
        "gpu_memory_mb": gpu_mb,
    }


@app.get("/health/gpu")
async def health_gpu() -> dict[str, str | bool | float | None]:
    """
    Debug endpoint: GPU utilization. Ensures model is on CUDA.
    Returns device name, allocated/reserved memory, compute capability.
    """
    info = get_gpu_info()
    if info is None:
        return {
            "status": "ok",
            "cuda_available": False,
            "model_loaded": is_loaded(),
            "message": "CUDA not available",
        }
    return {
        "status": "ok",
        "cuda_available": True,
        "model_loaded": is_loaded(),
        "device_name": info["device_name"],
        "allocated_mb": info["allocated_mb"],
        "reserved_mb": info["reserved_mb"],
        "compute_capability": info["compute_capability"],
    }


@app.exception_handler(HTTPException)
async def http_exception_handler(request, exc: HTTPException):
    """Ensure HTTP exceptions return structured JSON with request_id when possible."""
    rid = getattr(request.state, "request_id", None) or generate_request_id()
    return JSONResponse(
        status_code=exc.status_code,
        content=error_payload("http_error", str(exc.detail), rid),
    )
