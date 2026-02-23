"""
Application configuration. Pydantic v2 settings. SDXL image service.
"""

from __future__ import annotations

from functools import lru_cache
from pathlib import Path
from typing import Literal

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
        env_ignore_empty=True,
        protected_namespaces=("settings_",),
    )

    app_name: str = Field(default="SDXL Image Service")
    debug: bool = Field(default=False)
    environment: Literal["development", "staging", "production"] = Field(default="development")

    host: str = Field(default="0.0.0.0")
    port: int = Field(default=7000, ge=1, le=65535)
    base_path: str = Field(default="")

    static_dir: Path = Field(default=Path("static"))
    generated_dir_name: str = Field(default="generated")
    frontend_dir: Path | None = Field(default=Path("static_frontend"))
    upload_max_size_mb: int = Field(default=20, ge=1, le=100)
    training_dir_name: str = Field(default="data/training")
    lora_adapters_dir: str = Field(default="data/lora", description="LoRA safetensors")

    # Z-Image-Turbo (Hugging Face에서 다운로드 후 로컬에서 추론)
    model_id: str = Field(
        default="Tongyi-MAI/Z-Image-Turbo",
        description="Hugging Face model ID for Z-Image img2img (로컬에서 실행)",
    )
    # SDXL models (commercial-safe: Stability AI SDXL 1.0, Animagine XL)
    sdxl_base_id: str = Field(default="stabilityai/stable-diffusion-xl-base-1.0")
    animagine_xl_id: str = Field(default="cagliostrolab/animagine-xl-3.0")

    device: Literal["cuda", "cpu", "auto"] = Field(default="auto")
    torch_dtype: Literal["float16", "bfloat16"] = Field(default="bfloat16")
    max_inference_steps: int = Field(default=50, ge=1, le=100)
    min_inference_steps: int = Field(default=1, ge=1, le=50)
    max_resolution: int = Field(default=1024, ge=512, le=2048)
    min_resolution: int = Field(default=512, ge=256, le=1024)
    default_cfg: float = Field(default=7.5, ge=1.0, le=20.0)
    default_strength: float = Field(default=0.75, ge=0.0, le=1.0)

    enable_xformers: bool = Field(default=True)
    enable_tf32: bool = Field(default=True)
    enable_vae_slicing: bool = Field(default=True)
    enable_attention_slicing: bool = Field(default=True)
    enable_torch_compile: bool = Field(default=False, description="Set True when safe for your GPU")

    gpu_semaphore_limit: int = Field(default=2, ge=1, le=16, description="Max concurrent GPU inferences")

    cors_origins: str = Field(default="*")
    cors_allow_credentials: bool = Field(default=True)
    rate_limit_requests: int = Field(default=100, ge=1, le=500)
    rate_limit_window_seconds: int = Field(default=60, ge=1, le=3600)

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
    def upload_max_bytes(self) -> int:
        return self.upload_max_size_mb * 1024 * 1024


@lru_cache
def get_settings() -> Settings:
    return Settings()
