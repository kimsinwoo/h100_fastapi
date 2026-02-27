"""
HunyuanImage-3.0-Instruct FastAPI application.
Single endpoint: POST /edit. Model loaded once at startup (singleton).
"""

from __future__ import annotations

import asyncio
import io
import logging
import random
import time
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Annotated

from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, StreamingResponse

from app.config import get_settings
from app.model import _assert_cuda_and_log, generate_image, is_loaded, load_model
from app.schemas import error_payload
from app.utils import generate_request_id, get_gpu_memory_mb, get_gpu_info, save_upload_to_temp_file

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger(__name__)

_gpu_semaphore: asyncio.Semaphore | None = None


def _get_semaphore() -> asyncio.Semaphore:
    if _gpu_semaphore is None:
        raise RuntimeError("Semaphore not initialized")
    return _gpu_semaphore


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Startup: validate CUDA, then load HunyuanImage-3.0-Instruct once. Shutdown: log."""
    global _gpu_semaphore
    # Fail fast if no GPU (crash startup)
    await asyncio.to_thread(_assert_cuda_and_log)
    settings = get_settings()
    _gpu_semaphore = asyncio.Semaphore(settings.MAX_CONCURRENT_JOBS)
    logger.info("MAX_CONCURRENT_JOBS=%s TIMEOUT_SECONDS=%s", settings.MAX_CONCURRENT_JOBS, settings.TIMEOUT_SECONDS)
    await asyncio.to_thread(load_model)
    logger.info("Model loaded: %s", is_loaded())
    yield
    logger.info("Shutting down")


app = FastAPI(
    title="HunyuanImage-3.0-Instruct API",
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
    steps: Annotated[int | None, Form()] = None,
    resolution: Annotated[str | None, Form()] = None,
) -> StreamingResponse | JSONResponse:
    """
    Edit the uploaded image according to the text prompt.
    Returns PNG binary stream (no base64). Max 1 image for this endpoint.
    """
    request_id = generate_request_id()
    start_time = time.perf_counter()

    if not prompt or not prompt.strip():
        return JSONResponse(
            status_code=400,
            content=error_payload("BadRequest", "Missing or empty prompt", request_id),
        )
    if not image.content_type or not image.content_type.startswith("image/"):
        return JSONResponse(
            status_code=400,
            content=error_payload("BadRequest", "Invalid input image: file must be an image", request_id),
        )

    temp_path: Path | None = None
    try:
        temp_path = await save_upload_to_temp_file(image)
    except ValueError as e:
        return JSONResponse(
            status_code=400,
            content=error_payload("BadRequest", str(e), request_id),
        )

    settings = get_settings()
    seed_val = seed if seed is not None else random.randint(0, 2**31 - 1)
    steps_val = steps if steps is not None else settings.DEFAULT_STEPS
    resolution_val = resolution or settings.DEFAULT_IMAGE_SIZE
    image_paths = [str(temp_path)]

    sem = _get_semaphore()
    try:
        await asyncio.wait_for(sem.acquire(), timeout=float(settings.TIMEOUT_SECONDS))
    except asyncio.TimeoutError:
        if temp_path and temp_path.exists():
            temp_path.unlink(missing_ok=True)
        return JSONResponse(
            status_code=504,
            content=error_payload("Timeout", "Request timed out waiting for GPU", request_id),
        )

    try:
        try:
            result_pil = await asyncio.wait_for(
                asyncio.to_thread(
                    generate_image,
                    prompt=prompt.strip(),
                    image_paths=image_paths,
                    seed=seed_val,
                    image_size=resolution_val,
                    use_system_prompt=settings.DEFAULT_USE_SYSTEM_PROMPT,
                    bot_task=settings.DEFAULT_BOT_TASK,
                    infer_align_image_size=True,
                    diff_infer_steps=steps_val,
                    verbose=1,
                    request_id=request_id,
                ),
                timeout=float(settings.TIMEOUT_SECONDS),
            )
        except asyncio.TimeoutError:
            return JSONResponse(
                status_code=504,
                content=error_payload("Timeout", "Inference timed out", request_id),
            )
        except RuntimeError as e:
            err_msg = str(e)
            if "out of memory" in err_msg.lower() or "oom" in err_msg.lower():
                return JSONResponse(
                    status_code=503,
                    content=error_payload("ServiceUnavailable", f"GPU out of memory: {err_msg}", request_id),
                )
            return JSONResponse(
                status_code=503,
                content=error_payload("ServiceUnavailable", err_msg, request_id),
            )
        except Exception as e:
            logger.exception("Inference failed request_id=%s", request_id)
            return JSONResponse(
                status_code=503,
                content=error_payload("ServiceUnavailable", str(e), request_id),
            )
        finally:
            if temp_path is not None and temp_path.exists():
                temp_path.unlink(missing_ok=True)

        buf = io.BytesIO()
        result_pil.save(buf, format="PNG")
        buf.seek(0)
        total_duration = time.perf_counter() - start_time
        logger.info("request_id=%s total_sec=%.2f seed=%s", request_id, total_duration, seed_val)

        return StreamingResponse(
            buf,
            media_type="image/png",
            headers={"X-Request-Id": request_id},
        )
    finally:
        sem.release()


@app.get("/health")
async def health() -> dict[str, str | bool | float | None]:
    """Health check: status, model loaded, GPU memory."""
    gpu_mb = get_gpu_memory_mb()
    return {
        "status": "ok",
        "model_loaded": is_loaded(),
        "gpu_memory_mb": gpu_mb,
    }


@app.get("/health/gpu")
async def health_gpu() -> dict[str, str | bool | float | None]:
    """Debug: GPU utilization, device name, allocated/reserved memory, compute capability."""
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
    rid = getattr(request.state, "request_id", None) or generate_request_id()
    return JSONResponse(
        status_code=exc.status_code,
        content=error_payload("HttpError", str(exc.detail), rid),
    )
