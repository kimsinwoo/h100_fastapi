"""
Production FastAPI app: lifespan (client init, graceful shutdown), logging, no blocking.
"""

from __future__ import annotations

import asyncio
import logging
import signal
import sys
from contextlib import asynccontextmanager
from typing import AsyncIterator

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from vllm_server.config import get_vllm_settings
from vllm_server.routes import router
from vllm_server.service import get_vllm_service

logging.basicConfig(
    level=getattr(logging, get_vllm_settings().log_level),
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
    stream=sys.stdout,
)
logger = logging.getLogger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncIterator[None]:
    """Startup: ensure service client. Shutdown: close client, cancel in-flight."""
    settings = get_vllm_settings()
    mode = "mock" if settings.use_mock else (settings.backend_url or "embedded")
    logger.info(
        "vLLM gateway starting: backend=%s, max_concurrent=%s",
        mode,
        settings.max_concurrent_requests,
    )
    yield
    logger.info("vLLM gateway shutting down...")
    try:
        await get_vllm_service().close()
    except Exception as e:
        logger.warning("Error closing vLLM service: %s", e)
    logger.info("vLLM gateway shutdown complete.")


def create_app() -> FastAPI:
    app = FastAPI(
        title="vLLM Chat API",
        version="1.0.0",
        lifespan=lifespan,
        docs_url="/docs",
        redoc_url="/redoc",
    )
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )
    app.include_router(router)
    return app


app = create_app()


def _handle_sigterm(signum: int, frame: object) -> None:
    """Graceful shutdown on SIGTERM/SIGINT."""
    logger.info("Received signal %s, initiating shutdown.", signum)
    raise SystemExit(0)


if __name__ == "__main__":
    import uvicorn
    settings = get_vllm_settings()
    signal.signal(signal.SIGTERM, _handle_sigterm)
    signal.signal(signal.SIGINT, _handle_sigterm)
    uvicorn.run(
        "vllm_server.main:app",
        host=settings.host,
        port=settings.port,
        workers=1,
        log_level=settings.log_level.lower(),
    )
