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
    port: int = Field(default=7000, ge=1, le=65535)
    # 리버스 프록시 등으로 앱이 하위 경로에 있을 때 (예: /95ce287337c3ad9f). 비우면 루트.
    base_path: str = Field(default="", description="예: /95ce287337c3ad9f — 이 경로 아래로 앱 마운트")

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

    # Rate limiting (동시 100명 사용 시 100~200 권장)
    rate_limit_requests: int = Field(default=100, ge=1, le=500)
    rate_limit_window_seconds: int = Field(default=60, ge=1, le=3600)

    # LLM: 로컬에서 gpt-oss-20b(transformers) 사용
    llm_enabled: bool = Field(default=True, description="LLM 사용 여부")
    llm_use_local: bool = Field(
        default=True,
        description="True면 로컬에서 모델 로드·추론(transformers). False면 llm_api_base 호출.",
    )
    llm_local_model_id: str = Field(
        default="openai/gpt-oss-20b",
        description="로컬 사용 시 Hugging Face 모델 ID (gpt-oss-20b). 게이트면 LLM_HF_TOKEN 설정.",
    )
    llm_hf_token: str = Field(default="", description="게이트 모델일 때 Hugging Face 토큰.")
    llm_api_base: str = Field(default="", description="llm_use_local=False일 때만. OpenAI 호환 API 베이스 URL.")
    llm_model: str = Field(
        default="openai/gpt-oss-20b",
        description="API 사용 시 모델 이름. vLLM 서빙명과 일치해야 함(예: openai/gpt-oss-20b).",
    )
    llm_api_key: str = Field(default="", description="외부 API 사용 시에만 필요")
    llm_timeout_seconds: int = Field(default=120, ge=10, le=600)
    llm_max_concurrent: int = Field(
        default=1,
        ge=1,
        le=100,
        description="동시 LLM 요청 수. 로컬 1GPU는 반드시 1 (한 번에 한 추론만).",
    )
    llm_queue_wait_seconds: int = Field(
        default=20,
        ge=5,
        le=120,
        description="로컬 LLM 대기 시간(초). 앞 사용자 처리 중이면 이 시간 초과 시 503.",
    )
    # H100 등에서 추론 속도 향상: flash_attention_2 > sdpa > eager. True면 시도 후 없으면 sdpa/eager
    llm_use_flash_attention: bool = Field(default=True, description="True면 Flash Attention 2 또는 SDPA 사용 시도")
    # LLM 로드 후 기동 시 한국어 LoRA 파인튜닝 백그라운드 실행 여부 (기본 끔: 채팅과 리소스 경쟁 방지)
    llm_lora_finetune_on_startup: bool = Field(
        default=False,
        description="True면 서버 기동·LLM 로드 후 백그라운드에서 한국어 LoRA 파인튜닝을 시작. 채팅 안정을 위해 기본 False.",
    )

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
    def korean_lora_output_dir(self) -> Path:
        """한국어 LoRA 파인튜닝 결과 저장 경로 (training_dir/korean_lora)."""
        return self.training_dir / "korean_lora"

    @property
    def chat_rooms_dir(self) -> Path:
        """채팅방 저장 디렉터리 (backend/data/chats)."""
        from pathlib import Path as P
        backend_dir = P(__file__).resolve().parent.parent.parent
        return backend_dir / "data" / "chats"

    @property
    def upload_max_bytes(self) -> int:
        return self.upload_max_size_mb * 1024 * 1024


@lru_cache
def get_settings() -> Settings:
    return Settings()
