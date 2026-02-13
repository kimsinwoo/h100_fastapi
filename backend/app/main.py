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
from fastapi.responses import FileResponse
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

    # API와 /health 먼저 등록 (아래 프론트엔드 마운트보다 우선)
    app.include_router(router)

    @app.get("/health")
    async def health() -> dict[str, str | bool]:
        return {
            "status": "ok",
            "gpu_available": is_gpu_available(),
        }

    # 생성 이미지용 static (API 이미지 URL이 /static/generated/... 로 오므로 먼저 마운트)
    static_path = Path(settings.static_dir)
    static_path.mkdir(parents=True, exist_ok=True)
    app.mount("/static", StaticFiles(directory=str(static_path)), name="static")

    # 빌드된 프론트엔드 서빙: static_frontend 폴더가 있으면 / 에서 index.html + /assets/*.js 제공
    # (이렇게 해야 JS 요청에 HTML이 아닌 실제 .js가 가서 MIME 오류가 나지 않음)
    frontend_path = settings.frontend_dir if settings.frontend_dir else Path("static_frontend")
    if frontend_path.is_absolute():
        frontend_resolved = frontend_path
    else:
        # backend/app/main.py -> parent.parent = backend 디렉터리
        backend_dir = Path(__file__).resolve().parent.parent
        frontend_resolved = backend_dir / frontend_path
    # 배포용 빌드만 서빙: index.html 이 /assets/ 를 참조해야 함.
    index_path = frontend_resolved / "index.html"
    frontend_ok = False
    if frontend_resolved.exists() and index_path.exists():
        index_content = index_path.read_text(encoding="utf-8", errors="ignore")
        if "/assets/" in index_content and "main.tsx" not in index_content and "react-refresh" not in index_content:
            assets_dir = frontend_resolved / "assets"
            if assets_dir.is_dir():
                app.mount("/assets", StaticFiles(directory=str(assets_dir)), name="frontend_assets")
            else:
                logger.warning("Frontend assets dir not found: %s — check build output", assets_dir)

            @app.get("/")
            async def serve_index():
                return FileResponse(index_path, media_type="text/html")

            @app.get("/{path:path}")
            async def spa_fallback(path: str):
                # /api, /static, /health, /docs, /assets 는 위에서 처리됨. 나머지 GET 은 index.html (SPA)
                return FileResponse(index_path, media_type="text/html")

            logger.info("Serving frontend (production build) from %s", frontend_resolved)
            frontend_ok = True
        else:
            logger.warning(
                "Frontend at %s looks like DEV build. Run: ./scripts/build_and_serve.sh then restart.",
                frontend_resolved,
            )
    if not frontend_ok:
        @app.get("/")
        async def root_help():
            from fastapi.responses import HTMLResponse
            return HTMLResponse(
                "<!DOCTYPE html><html><head><meta charset='utf-8'><title>Z-Image AI</title></head><body>"
                "<h1>Z-Image AI</h1><p>API 문서: <a href='/docs'>/docs</a></p>"
                "<p><strong>프론트가 안 뜨면:</strong> <code>./scripts/build_and_serve.sh</code> 실행 후 서버 재시작.</p>"
                "<p>경로 확인: <code>%s</code></p></body></html>"
                % (frontend_resolved,),
                status_code=200,
            )
        if not frontend_resolved.exists():
            logger.warning("Frontend dir not found: %s — run scripts/build_and_serve.sh", frontend_resolved)

    return app


app = create_app()
