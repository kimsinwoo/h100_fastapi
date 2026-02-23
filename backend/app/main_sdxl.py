"""
SDXL Image Service — production FastAPI. H100-optimized, LoRA, txt2img/img2img.
"""

from __future__ import annotations

import os
import logging
from pathlib import Path

# xformers + Triton/PyTorch 2.9 호환 오류(JITCallable._set_src) 방지: 실제 xformers 로드 차단
os.environ.setdefault("PYTORCH_JIT", "0")
os.environ.setdefault("TORCH_COMPILE_DISABLE", "1")
import sys
from types import ModuleType


class _XformersOpsFake(ModuleType):
    def __getattr__(self, name):
        raise ImportError("xformers disabled for Triton/PyTorch compatibility (JITCallable._set_src)")


_xops = _XformersOpsFake("xformers.ops")
_xformers = ModuleType("xformers")
_xformers.ops = _xops
sys.modules["xformers"] = _xformers
sys.modules["xformers.ops"] = _xops

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
