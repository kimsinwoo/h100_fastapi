"""
Application configuration. Pydantic v2 settings. Z-Image-Turbo 전용.
"""

from __future__ import annotations

from functools import lru_cache
from pathlib import Path
from typing import Literal

from pydantic import Field, model_validator
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
        env_ignore_empty=True,
        protected_namespaces=("settings_",),
    )

    app_name: str = Field(default="Z-Image-Turbo Service")
    debug: bool = Field(default=False)
    environment: Literal["development", "staging", "production"] = Field(default="development")

    host: str = Field(default="0.0.0.0")
    port: int = Field(default=7000, ge=1, le=65535)
    base_path: str = Field(
        default="95ce287337c3ad9f",
        description="리버스 프록시 하위 경로. 프론트 baseURL과 맞추면 /base_path/api/generate 로 요청 처리.",
    )

    static_dir: Path = Field(default=Path("static"))
    generated_dir_name: str = Field(default="generated")
    frontend_dir: Path | None = Field(default=Path("static_frontend"))
    upload_max_size_mb: int = Field(default=20, ge=1, le=100)
    training_dir_name: str = Field(default="data/training")
    lora_adapters_dir: str = Field(default="data/lora", description="LoRA safetensors")
    chat_rooms_dir_name: str = Field(
        default="data/chat_rooms",
        description="채팅방 JSON 저장 디렉터리(backend_dir 기준 상대 경로).",
    )

    # 이미지 생성 백엔드: OmniGen(Omni) 사용 시 H100 등에서 Omni 모델 로드
    use_omnigen: bool = Field(
        default=True,
        description="True면 OmniGen(Omni) 파이프라인 사용, False면 Z-Image-Turbo 사용 (환경변수 USE_OMNIGEN)",
    )
    omnigen_model_id: str = Field(
        default="Shitao/OmniGen-v1-diffusers",
        description="OmniGen Hugging Face model ID (use_omnigen=True일 때)",
    )

    # Z-Image-Turbo (use_omnigen=False일 때만 사용)
    model_id: str = Field(
        default="Tongyi-MAI/Z-Image-Turbo",
        description="Hugging Face model ID for Z-Image img2img",
    )

    device: Literal["cuda", "cpu", "auto"] = Field(default="auto")
    torch_dtype: Literal["float16", "bfloat16"] = Field(default="bfloat16")
    max_inference_steps: int = Field(default=50, ge=1, le=100)
    min_inference_steps: int = Field(default=1, ge=1, le=50)
    max_resolution: int = Field(default=1024, ge=512, le=2048)
    min_resolution: int = Field(default=512, ge=256, le=1024)
    default_cfg: float = Field(default=7.5, ge=1.0, le=20.0)
    default_strength: float = Field(default=0.75, ge=0.0, le=1.0)

    enable_xformers: bool = Field(default=True, description="Z-Image 전용. H100 등에서 메모리 효율 어텐션 (속도 향상)")
    enable_tf32: bool = Field(default=True)
    enable_vae_slicing: bool = Field(default=False, description="H100 등 VRAM 넉넉하면 False가 더 빠름. True면 VRAM 절약")
    enable_attention_slicing: bool = Field(default=False, description="H100 등에서는 False가 더 빠름. True면 VRAM 절약")
    enable_vae_tiling: bool = Field(default=False, description="대형 해상도 시 VRAM 절약, 보통 비활성화")
    enable_flash_attention_2: bool = Field(default=True, description="OmniGen 로드 시 attn_implementation=flash_attention_2 시도 (flash-attn 필요)")
    omnigen_max_input_size: int = Field(default=1024, ge=512, le=1024, description="OmniGen 입력 최대 변. HF와 동일하게 1024 기본")
    enable_torch_compile: bool = Field(default=False, description="Set True when safe for your GPU")

    gpu_semaphore_limit: int = Field(default=2, ge=1, le=16, description="Max concurrent GPU inferences")

    # LLM: Qwen3.5-35B-A3B (텍스트/멀티모달 채팅·프롬프트 추천)
    # 다중 사용자: LLM_USE_VLLM=true 로 두면 vLLM(7001) 사용. 단일 사용자/개발은 로컬(transformers) 기본.
    llm_use_vllm: bool = Field(
        default=False,
        description="True면 다중 사용자용 vLLM API 사용(LLM_USE_LOCAL=false, LLM_API_BASE 기본 7001/v1)",
    )
    llm_enabled: bool = Field(default=True, description="LLM 채팅/프롬프트 추천 사용 여부")
    llm_use_local: bool = Field(
        default=True,
        description="True면 transformers로 로컬 로드, False면 llm_api_base 호출",
    )
    llm_local_model_id: str = Field(
        default="Qwen/Qwen3.5-35B-A3B",
        description="로컬 LLM Hugging Face 모델 ID (기본: Qwen3.5-35B-A3B)",
    )
    llm_model: str = Field(
        default="Qwen/Qwen3.5-35B-A3B",
        description="API 모드일 때 요청에 넣을 model 이름",
    )
    llm_api_base: str = Field(
        default="",
        description="OpenAI 호환 API 베이스 URL (예: http://localhost:8000/v1). 비면 로컬만 사용.",
    )
    llm_api_key: str = Field(default="", description="API 키 (Bearer)")
    llm_max_concurrent: int = Field(default=4, ge=1, le=32)
    llm_queue_wait_seconds: float = Field(default=60.0, ge=5.0, le=300.0)
    llm_timeout_seconds: float = Field(default=120.0, ge=30.0, le=600.0)
    llm_use_flash_attention: bool = Field(default=True, description="로컬 로드 시 flash_attention_2/sdpa 시도")
    llm_hf_token: str = Field(default="", description="게이트 모델용 Hugging Face 토큰")
    korean_lora_output_dir: str = Field(
        default="korean_lora_output",
        description="한국어 LoRA 어댑터 디렉터리(backend_dir 기준 상대 경로)",
    )

    cors_origins: str = Field(default="*")
    cors_allow_credentials: bool = Field(default=True)
    rate_limit_requests: int = Field(default=100, ge=1, le=500)
    rate_limit_window_seconds: int = Field(default=60, ge=1, le=3600)

    @model_validator(mode="before")
    @classmethod
    def _apply_vllm_multi_user(cls, data: dict | object) -> dict | object:
        """다중 사용자: LLM_USE_VLLM=true 이면 vLLM API 사용(로컬 비활성화, api_base 기본값)."""
        if isinstance(data, dict) and data.get("llm_use_vllm"):
            updates = {"llm_use_local": False}
            if not (data.get("llm_api_base") or "").strip():
                updates["llm_api_base"] = "http://127.0.0.1:7001/v1"
            data = {**data, **updates}
        return data

    @property
    def backend_dir(self) -> Path:
        return Path(__file__).resolve().parent.parent.parent

    @property
    def generated_dir(self) -> Path:
        base = self.static_dir if self.static_dir.is_absolute() else self.backend_dir / self.static_dir
        return base / self.generated_dir_name

    @property
    def training_dir(self) -> Path:
        return self.backend_dir / self.training_dir_name

    @property
    def lora_dir(self) -> Path:
        p = self.backend_dir / self.lora_adapters_dir
        p.mkdir(parents=True, exist_ok=True)
        return p

    @property
    def chat_rooms_dir(self) -> Path:
        """채팅방 JSON 저장 루트 (user_id별 하위 디렉터리)."""
        p = self.backend_dir / self.chat_rooms_dir_name
        p.mkdir(parents=True, exist_ok=True)
        return p

    # LoRA 학습 결과 (train_lora_zit.py 출력) — 스타일별 추론 시 여기서 로드
    @property
    def lora_output_dir(self) -> Path:
        return self.backend_dir / "lora_output"

    @property
    def korean_lora_output_dir_path(self) -> Path:
        """한국어 LLM LoRA 어댑터 경로 (adapter_config.json 위치)."""
        return self.backend_dir / self.korean_lora_output_dir

    @property
    def upload_max_bytes(self) -> int:
        return self.upload_max_size_mb * 1024 * 1024


@lru_cache
def get_settings() -> Settings:
    return Settings()
