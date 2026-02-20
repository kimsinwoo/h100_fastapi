"""
Z-Image AI Service — production FastAPI application.
"""

from __future__ import annotations

import asyncio
import logging
import subprocess
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
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles

from app.api import router
from app.core.config import get_settings
from app.services.image_service import get_pipeline, is_gpu_available, is_pipeline_loaded
from app.services.llm_service import preload_local_model
from app.utils.file_handler import ensure_generated_dir

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger(__name__)


def _start_lora_finetune_background(settings) -> None:
    """LLM 로드 후 한국어 LoRA 파인튜닝을 서브프로세스로 백그라운드 실행. 서버는 블로킹 없이 기동."""
    backend_dir = Path(__file__).resolve().parent.parent
    scripts_dir = backend_dir.parent / "scripts" / "korean_lora"
    train_script = scripts_dir / "train_lora.py"
    data_file = scripts_dir / "data" / "korean_sft_train.jsonl"
    build_script = scripts_dir / "data" / "build_korean_sft.py"
    out_dir = settings.korean_lora_output_dir
    out_dir.mkdir(parents=True, exist_ok=True)

    if not train_script.exists():
        logger.warning("LoRA 학습 스크립트 없음: %s — 파인튜닝 생략", train_script)
        return
    # 학습 데이터 없으면 생성; 있으면 최신 예시 반영을 위해 매번 재생성 (건강 도우미 예시 포함)
    if build_script.exists():
        try:
            subprocess.run(
                [sys.executable, str(build_script)],
                cwd=str(scripts_dir),
                capture_output=True,
                timeout=90,
                check=True,
            )
            logger.info("한국어 SFT 데이터 준비 완료: %s", data_file)
        except (subprocess.CalledProcessError, subprocess.TimeoutExpired) as e:
            logger.warning("SFT 데이터 생성 실패: %s", e)
    if not data_file.exists():
        logger.warning("학습 데이터 없음: %s — 파인튜닝 생략", data_file)
        return

    cmd = [
        sys.executable,
        str(train_script),
        "--model_name_or_path", settings.llm_local_model_id,
        "--output_dir", str(out_dir),
        "--data_file", str(data_file),
        "--num_epochs", "5",
        "--per_device_train_batch_size", "2",
    ]
    if settings.llm_hf_token and settings.llm_hf_token.strip():
        cmd.extend(["--hf_token", settings.llm_hf_token.strip()])

    async def _run() -> None:
        try:
            proc = await asyncio.create_subprocess_exec(
                *cmd,
                cwd=str(scripts_dir),
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
            )
            stdout, stderr = await proc.communicate()
            if proc.returncode == 0:
                logger.info("한국어 LoRA 파인튜닝 완료: %s", out_dir)
            else:
                logger.warning("한국어 LoRA 파인튜닝 종료 코드 %s. stderr: %s", proc.returncode, stderr.decode(errors="replace")[:500])
        except Exception as e:
            logger.warning("한국어 LoRA 파인튜닝 실행 중 오류: %s", e)

    asyncio.create_task(_run())
    logger.info("한국어 LoRA 파인튜닝 백그라운드 시작 (출력: %s)", out_dir)


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Startup: ensure static dir, load model. Shutdown: cleanup."""
    settings = get_settings()
    ensure_generated_dir()
    logger.info("Static directory ready: %s", settings.generated_dir)
    # Load image model once at startup
    try:
        await get_pipeline()
        logger.info("GPU available: %s", is_gpu_available())
    except Exception as e:
        logger.warning("Model not loaded at startup (will fail on first request): %s", e)
    # LLM 로컬 사용 시 기동 시 미리 로드 (첫 채팅 요청 대기 없음)
    try:
        await preload_local_model()
    except Exception as e:
        logger.warning("LLM preload at startup failed: %s", e)

    # LLM 로드 후 한국어 LoRA 파인튜닝을 백그라운드에서 시작 (설정 시)
    if (
        settings.llm_enabled
        and settings.llm_use_local
        and getattr(settings, "llm_lora_finetune_on_startup", False)
    ):
        _start_lora_finetune_background(settings)

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
            "model_loaded": is_pipeline_loaded(),
        }

    @app.get("/api/info")
    async def api_info() -> dict[str, str | bool]:
        """이미지 생성 모델이 Hugging Face에서 다운로드 후 이 컴퓨터에서만 실행되는지 확인."""
        return {
            "image_model_source": "Hugging Face (Tongyi-MAI/Z-Image-Turbo)",
            "runs_locally": True,
            "model_loaded": is_pipeline_loaded(),
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
                "<h1>Z-Image AI</h1><p>API 문서: <a href='/docs'>/docs</a></p>"
                "<p><strong>프론트가 안 뜨면:</strong> <code>./scripts/build_and_serve.sh</code> 실행 후 서버 재시작.</p>"
                "<p>경로 확인: <code>%s</code></p></body></html>"
                % (frontend_resolved,),
                status_code=200,
            )
        if not frontend_resolved.exists():
            logger.warning("Frontend dir not found: %s — run scripts/build_and_serve.sh", frontend_resolved)

    # SPA fallback: 404 시에만 index.html. /assets, /api, /static 은 절대 HTML 로 폴백하지 않음 (MIME 오류 방지)
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
    """base_path 설정 시 해당 경로 아래로 앱 마운트 (리버스 프록시 대응)."""
    _app = create_app()
    base = (get_settings().base_path or "").strip().strip("/")
    if not base:
        return _app
    root = FastAPI(title="Z-Image AI (root)")
    root.mount(f"/{base}", _app)
    logger.info("App mounted at /%s", base)
    return root


app = _get_app()
