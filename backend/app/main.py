"""
Z-Image AI Service — production FastAPI application.
"""

from __future__ import annotations

import logging
import time
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles

from app.api import router
from app.core.config import get_settings
from app.services.image_service import get_pipeline, is_gpu_available
from app.utils.file_handler import ensure_generated_dir

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Startup: ensure static dir, load model. Shutdown: cleanup."""
    settings = get_settings()
    ensure_generated_dir()
    logger.info("Static directory ready: %s", settings.generated_dir)
    # Load model once at startup
    try:
        await get_pipeline()
        logger.info("GPU available: %s", is_gpu_available())
    except Exception as e:
        logger.warning("Model not loaded at startup (will fail on first request): %s", e)
    yield
    logger.info("Shutting down")


def create_app() -> FastAPI:
    settings = get_settings()
    app = FastAPI(
        title=settings.app_name,
        version="1.0.0",
        lifespan=lifespan,
    )

    # CORS
    origins = [o.strip() for o in settings.cors_origins.split(",") if o.strip()]
    if "*" in origins:
        origins = ["*"]
    app.add_middleware(
        CORSMiddleware,
        allow_origins=origins,
        allow_credentials=settings.cors_allow_credentials,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    # Logging middleware
    @app.middleware("http")
    async def log_requests(request: Request, call_next):
        start = time.perf_counter()
        response = await call_next(request)
        duration = time.perf_counter() - start
        logger.info("%s %s %s %.3fs", request.method, request.url.path, response.status_code, duration)
        return response

    # Mount static files for generated images
    static_path = Path(settings.static_dir)
    static_path.mkdir(parents=True, exist_ok=True)
    app.mount("/static", StaticFiles(directory=str(static_path)), name="static")

    app.include_router(router)

    @app.get("/health")
    async def health() -> dict[str, str | bool]:
        return {
            "status": "ok",
            "gpu_available": is_gpu_available(),
        }

    return app


app = create_app()
