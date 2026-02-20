"""
Application configuration. All settings from environment with sensible defaults.
"""

from __future__ import annotations

from functools import lru_cache
from pathlib import Path
from typing import Literal

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    # .env 없어도 기본값으로 동작. 있으면 로드해서 덮어씀.
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
        env_ignore_empty=True,
        protected_namespaces=("settings_",),  # model_id 등 model_ 접두어 필드명 허용
    )

    # App
    app_name: str = Field(default="Z-Image AI Service")
    debug: bool = Field(default=False)
    environment: Literal["development", "staging", "production"] = Field(default="development")

    # Server
    host: str = Field(default="0.0.0.0")
    port: int = Field(default=8000, ge=1, le=65535)

    # Paths
    static_dir: Path = Field(default=Path("static"))
    generated_dir_name: str = Field(default="generated")
    frontend_dir: Path | None = Field(default=Path("static_frontend"), description="빌드된 프론트엔드 폴더; 있으면 / 에서 서빙")
    upload_max_size_mb: int = Field(default=20, ge=1, le=100)
    # LoRA 학습용 데이터 저장 경로 (프로젝트 루트 기준 data/training)
    training_dir_name: str = Field(default="data/training", description="학습 이미지·메타데이터 저장 디렉터리")

    # Z-Image model
    model_id: str = Field(default="Tongyi-MAI/Z-Image-Turbo")
    device_preference: Literal["cuda", "cpu", "auto"] = Field(default="auto")
    default_strength: float = Field(default=0.6, ge=0.0, le=1.0)
    # FlowMatchEulerDiscreteScheduler: 9 steps 시 sigmas[10] 접근으로 IndexError 나는 이슈 있음 → 8로 설정
    default_steps: int = Field(default=8, ge=1, le=50)
    default_size: int = Field(default=1024, ge=512, le=2048)
    # MPS 메모리 부족 시 해상도 상한 (0이면 미적용). 768 권장.
    mps_max_size: int = Field(default=768, ge=0, le=2048, description="MPS 사용 시 최대 해상도 (0=제한 없음)")
    use_autocast: bool = Field(default=True)

    # CORS
    cors_origins: str = Field(default="*")
    cors_allow_credentials: bool = Field(default=True)

    # Rate limiting (동시 100명 사용 시 rate_limit_requests 100~200 권장)
    rate_limit_requests: int = Field(default=60, ge=1, le=500)
    rate_limit_window_seconds: int = Field(default=60, ge=1, le=3600)

    # LLM: 로컬 모델(transformers) 또는 외부 API
    llm_enabled: bool = Field(default=True, description="LLM 사용 여부")
    llm_use_local: bool = Field(
        default=True,
        description="True면 API 대신 로컬에서 모델 로드·추론. False면 llm_api_base 호출.",
    )
    llm_local_model_id: str = Field(
        default="google/gemma-2-2b-it",
        description="로컬 사용 시 Hugging Face 모델 ID (한국어·다국어 지원 추천: beomi/gemma-2-2b-it, Qwen/Qwen2-1.5B-Instruct 등)",
    )
    llm_api_base: str = Field(default="", description="로컬 미사용 시 OpenAI 호환 API 베이스 URL")
    llm_model: str = Field(default="gpt-oss-20b", description="외부 API 사용 시 모델 이름")
    llm_api_key: str = Field(default="", description="외부 API 사용 시에만 필요")
    llm_timeout_seconds: int = Field(default=120, ge=10, le=600)
    llm_max_concurrent: int = Field(default=20, ge=1, le=100, description="동시 LLM 요청 상한")

    @property
    def generated_dir(self) -> Path:
        return self.static_dir / self.generated_dir_name

    @property
    def training_dir(self) -> Path:
        """LoRA 학습용 데이터 디렉터리 (backend 기준 상대 경로)."""
        from pathlib import Path as P
        backend_dir = P(__file__).resolve().parent.parent.parent
        return backend_dir / self.training_dir_name

    @property
    def upload_max_bytes(self) -> int:
        return self.upload_max_size_mb * 1024 * 1024


@lru_cache
def get_settings() -> Settings:
    return Settings()
