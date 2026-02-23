"""
SDXL Image Service — production FastAPI. H100-optimized, LoRA, txt2img/img2img.
"""

from __future__ import annotations

import logging
from pathlib import Path

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles

from app.api.routes_sdxl import router as sdxl_router
from app.core.config import get_settings
from app.utils.gpu import init_gpu

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger(__name__)


def create_app() -> FastAPI:
    init_gpu()
    s = get_settings()
    app = FastAPI(title=s.app_name, version="1.0.0")
    app.add_middleware(
        CORSMiddleware,
        allow_origins=s.cors_origins.split(",") if s.cors_origins else ["*"],
        allow_credentials=s.cors_allow_credentials,
        allow_methods=["*"],
        allow_headers=["*"],
    )
    app.include_router(sdxl_router)

    static_path = s.backend_dir / "static"
    static_path.mkdir(parents=True, exist_ok=True)
    app.mount("/static", StaticFiles(directory=str(static_path)), name="static")

    @app.get("/")
    async def root():
        return {"service": "SDXL Image API", "docs": "/docs", "health": "/api/health"}

    return app


app = create_app()
