"""
Z-Image AI Service — production FastAPI application.
"""

from __future__ import annotations

import logging
import sys
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
from app.sd15.model_manager import is_model_loaded as sd15_model_loaded
from app.sd15.routes import router as sd15_router, run_sd15_startup
from app.utils.file_handler import ensure_generated_dir

try:
    from app.sdxl_prod.routes import router as sdxl_prod_router, run_sdxl_prod_startup
    SDXL_PROD_AVAILABLE = True
except Exception:
    SDXL_PROD_AVAILABLE = False
    sdxl_prod_router = None
    run_sdxl_prod_startup = None

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
    """Startup: static dir, SD 1.5 only (Hugging Face 또는 로컬). Z-Image/SDXL API는 배제."""
    settings = get_settings()
    import sys
    print(f"\n>>> SD 1.5 이미지 서버 기동: http://0.0.0.0:{settings.port} (API: /sd15, 상태: /health)\n", file=sys.stderr, flush=True)
    ensure_generated_dir()
    logger.info("Static directory ready: %s", settings.generated_dir)
    # SD 1.5만 사용 — Hugging Face에서 다운로드 후 로컬에서 추론 (로컬 model_path 없으면 model_id 사용)
    run_sd15_startup()
    # SDXL production (H100 multi-model): preload pipelines + worker queue
    if SDXL_PROD_AVAILABLE and run_sdxl_prod_startup is not None:
        run_sdxl_prod_startup()
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

    # SD 1.5 API + optional SDXL production (H100)
    app.include_router(sd15_router)
    if SDXL_PROD_AVAILABLE and sdxl_prod_router is not None:
        app.include_router(sdxl_prod_router)

    @app.exception_handler(RequestValidationError)
    async def validation_exception_handler(request: Request, exc: RequestValidationError):
        """422 검증 실패 시 원인 로그 (Content-Type·필드별 오류)."""
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
            "model_loaded": sd15_model_loaded(),
        }

    @app.get("/api/info")
    async def api_info() -> dict[str, str | bool]:
        """SD 1.5: Hugging Face에서 다운로드 후 이 컴퓨터에서만 실행."""
        return {
            "image_model_source": "Hugging Face (runwayml/stable-diffusion-v1-5)",
            "runs_locally": True,
            "model_loaded": sd15_model_loaded(),
        }

    # 생성 이미지용 static (API 이미지 URL이 /static/generated/... 로 오므로 먼저 마운트)
    static_path = Path(settings.static_dir)
    static_path.mkdir(parents=True, exist_ok=True)
    app.mount("/static", StaticFiles(directory=str(static_path)), name="static")

    # 빌드된 프론트엔드 서빙 (순서 중요: /assets 마운트 → GET / → 404 시에만 index.html)
    # Catch-all GET /{path:path} 사용 시 /assets/*.js 요청까지 index.html 로 가서 MIME 오류 발생하므로
    # /assets 는 StaticFiles 전용, SPA 폴백은 404 예외 핸들러로만 처리. 자세한 설명: docs/FASTAPI_VITE_STATIC_SERVING.md
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
                # /assets/* 는 반드시 여기서만 처리 → JS/CSS 가 MIME 맞게 내려감 (catch-all 이 가로채지 않음)
                app.mount("/assets", StaticFiles(directory=str(assets_dir)), name="frontend_assets")
            else:
                logger.warning("Frontend assets dir not found: %s — check build output", assets_dir)

            @app.get("/")
            async def serve_index():
                return FileResponse(index_path, media_type="text/html")

            # SPA: 404 일 때만 index.html 반환. catch-all 제거해서 /assets/xxx 가 HTML 로 안 나가게 함
            app.state.frontend_index_path = index_path

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
                "<h1>SD 1.5 Image API</h1><p>API 문서: <a href='/docs'>/docs</a>, 이미지 생성: <a href='/sd15/health'>/sd15</a></p>"
                "<p><strong>프론트가 안 뜨면:</strong> <code>./scripts/build_and_serve.sh</code> 실행 후 서버 재시작.</p>"
                "<p>경로 확인: <code>%s</code></p></body></html>"
                % (frontend_resolved,),
                status_code=200,
            )
        if not frontend_resolved.exists():
            logger.warning("Frontend dir not found: %s — run scripts/build_and_serve.sh", frontend_resolved)

    # SPA fallback: 404 시에만 index.html. /assets, /api, /static 은 절대 HTML 로 폴백하지 않음 (MIME 오류 방지)
    index_for_spa = getattr(app.state, "frontend_index_path", None)
    _NO_SPA_FALLBACK_PREFIXES = ("/assets", "/api", "/static", "/sd15", "/docs", "/openapi.json", "/redoc")

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
    """
    base_path 설정 시 해당 경로 아래로 앱 마운트.
    - 프록시가 접두사 제거 후 /health, /api/* 로 보내도 200 나오도록 루트에 동일 라우트·lifespan 등록.
    """
    _app = create_app()
    base = (get_settings().base_path or "").strip().strip("/")
    if not base:
        return _app
    # 루트도 동일 lifespan 사용 (파이프라인 로드 등) + CORS
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

    # 프록시가 /health, /sd15/*, /sdxl/* 만 보낼 때 404 방지: 루트에 동일 라우트 등록 (접두사 없이 호출 가능)
    root.include_router(sd15_router)
    if SDXL_PROD_AVAILABLE and sdxl_prod_router is not None:
        root.include_router(sdxl_prod_router)

    @root.get("/health")
    async def _health() -> dict[str, str | bool]:
        return {
            "status": "ok",
            "gpu_available": _gpu_available(),
            "model_loaded": sd15_model_loaded(),
        }

    root.mount(f"/{base}", _app)
    logger.info("App mounted at /%s (also serving /health and /sd15/* without prefix)", base)
    return root


app = _get_app()
