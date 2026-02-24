"""
Z-Image AI Service — production FastAPI application.
Z-Image-Turbo(img2img) 전용. /api/generate, /api/styles 등.
"""

from __future__ import annotations

import logging
import time
import warnings

# Mac 등 CUDA 미지원 환경에서 torch.amp가 'cuda' 사용 시 나는 경고 억제
warnings.filterwarnings(
    "ignore",
    message=".*device_type of 'cuda'.*CUDA is not available.*",
    category=UserWarning,
    module="torch.amp.autocast_mode",
)
from contextlib import asynccontextmanager
from pathlib import Path

from starlette.exceptions import HTTPException as StarletteHTTPException

from fastapi import FastAPI, Request
from fastapi.exceptions import RequestValidationError
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles

from app.core.config import get_settings
from app.api.routes import router as api_router
from app.utils.file_handler import ensure_generated_dir
from app.services.image_service import is_pipeline_loaded

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger(__name__)


def _gpu_available() -> bool:
    try:
        import torch
        return torch.cuda.is_available()
    except Exception:
        return False


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Startup: static dir, Z-Image-Turbo API만 사용. 파이프라인은 첫 /api/generate 시 lazy 로드."""
    settings = get_settings()
    ensure_generated_dir()
    logger.info("Static directory ready: %s", settings.generated_dir)
    logger.info("Z-Image-Turbo API: http://0.0.0.0:%s (API: /api/generate, /api/styles, 상태: /health)", settings.port)
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
        logger.info(" %s %s %s %.3fs", request.method, request.url.path, response.status_code, duration)
        return response

    app.include_router(api_router)

    @app.exception_handler(RequestValidationError)
    async def validation_exception_handler(request: Request, exc: RequestValidationError):
        ct = request.headers.get("content-type", "")
        logger.info(
            "422 Unprocessable Entity path=%s content_type=%s detail=%s",
            request.url.path,
            ct,
            exc.errors(),
        )
        return JSONResponse(status_code=422, content={"detail": exc.errors()})

    @app.get("/health")
    async def health() -> dict[str, str | bool]:
        return {
            "status": "ok",
            "gpu_available": _gpu_available(),
            "model_loaded": is_pipeline_loaded(),
        }

    @app.get("/api/info")
    async def api_info() -> dict[str, str | bool]:
        """Z-Image-Turbo: Hugging Face에서 다운로드 후 로컬에서 실행."""
        return {
            "image_model_source": "Z-Image-Turbo (Tongyi-MAI/Z-Image-Turbo)",
            "runs_locally": True,
            "model_loaded": is_pipeline_loaded(),
        }

    # 생성 이미지용 static
    static_path = Path(settings.static_dir)
    static_path.mkdir(parents=True, exist_ok=True)
    app.mount("/static", StaticFiles(directory=str(static_path)), name="static")

    # 빌드된 프론트엔드 서빙
    frontend_path = settings.frontend_dir if settings.frontend_dir else Path("static_frontend")
    if frontend_path.is_absolute():
        frontend_resolved = frontend_path
    else:
        backend_dir = Path(__file__).resolve().parent.parent
        frontend_resolved = backend_dir / frontend_path
    index_path = frontend_resolved / "index.html"
    frontend_ok = False
    if frontend_resolved.exists() and index_path.exists():
        index_content = index_path.read_text(encoding="utf-8", errors="ignore")
        if "/assets/" in index_content and "main.tsx" not in index_content and "react-refresh" not in index_content:
            assets_dir = frontend_resolved / "assets"
            if assets_dir.is_dir():
                app.mount("/assets", StaticFiles(directory=str(assets_dir)), name="frontend_assets")
            else:
                logger.warning("Frontend assets dir not found: %s", assets_dir)

            @app.get("/")
            async def serve_index():
                return FileResponse(index_path, media_type="text/html")

            app.state.frontend_index_path = index_path
            logger.info("Serving frontend from %s", frontend_resolved)
            frontend_ok = True
        else:
            logger.warning("Frontend at %s looks like DEV build. Run build then restart.", frontend_resolved)
    if not frontend_ok:
        @app.get("/")
        async def root_help():
            from fastapi.responses import HTMLResponse
            return HTMLResponse(
                "<!DOCTYPE html><html><head><meta charset='utf-8'><title>Z-Image AI</title></head><body>"
                "<h1>Z-Image-Turbo API</h1><p>API 문서: <a href='/docs'>/docs</a>, "
                "이미지 생성: <a href='/api/generate'>/api/generate</a>, 스타일: <a href='/api/styles'>/api/styles</a></p>"
                "<p><strong>프론트가 안 뜨면:</strong> <code>./scripts/build_and_serve.sh</code> 실행 후 서버 재시작.</p>"
                "<p>경로: <code>%s</code></p></body></html>"
                % (frontend_resolved,),
                status_code=200,
            )
        if not frontend_resolved.exists():
            logger.warning("Frontend dir not found: %s", frontend_resolved)

    index_for_spa = getattr(app.state, "frontend_index_path", None)
    _NO_SPA_FALLBACK_PREFIXES = ("/assets", "/api", "/static", "/docs", "/openapi.json", "/redoc")

    @app.exception_handler(StarletteHTTPException)
    async def http_exception_handler(request: Request, exc: StarletteHTTPException):
        if exc.status_code != 404:
            return JSONResponse(status_code=exc.status_code, content={"detail": exc.detail})
        if request.method != "GET" or index_for_spa is None:
            return JSONResponse(status_code=404, content={"detail": exc.detail})
        path = request.url.path
        if any(path.startswith(p) for p in _NO_SPA_FALLBACK_PREFIXES):
            return JSONResponse(status_code=404, content={"detail": "Not Found"})
        return FileResponse(index_for_spa, media_type="text/html")

    return app


def _get_app() -> FastAPI:
    """base_path 설정 시 해당 경로 아래로 앱 마운트."""
    _app = create_app()
    base = (get_settings().base_path or "").strip().strip("/")
    if not base:
        return _app
    settings = get_settings()
    origins = [o.strip() for o in settings.cors_origins.split(",") if o.strip()]
    if "*" in origins:
        origins = ["*"]
    root = FastAPI(title="Z-Image AI (root)", lifespan=lifespan)
    root.add_middleware(
        CORSMiddleware,
        allow_origins=origins,
        allow_credentials=settings.cors_allow_credentials,
        allow_methods=["*"],
        allow_headers=["*"],
    )
    root.include_router(api_router)

    @root.get("/health")
    async def _health() -> dict[str, str | bool]:
        return {
            "status": "ok",
            "gpu_available": _gpu_available(),
            "model_loaded": is_pipeline_loaded(),
        }

    root.mount(f"/{base}", _app)
    logger.info("App mounted at /%s (also serving /health, /api without prefix)", base)
    return root


app = _get_app()
