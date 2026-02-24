"""
Production SDXL config: H100 80GB, 20 concurrent users, 3 parallel GPU jobs.
"""
from __future__ import annotations

from pathlib import Path

from pydantic import Field, field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict


def _backend_dir() -> Path:
    return Path(__file__).resolve().parent.parent.parent


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
        env_prefix="SDXL_PROD_",
    )

    device: str = Field(default="cuda")
    torch_dtype: str = Field(default="float16")
    enable_tf32: bool = Field(default=True)
    enable_xformers: bool = Field(default=True)
    enable_vae_slicing: bool = Field(default=True)
    enable_attention_slicing: bool = Field(default=True)
    enable_torch_compile: bool = Field(default=False, description="Set True when stable on your setup")
    safety_checker: bool = Field(default=False)

    max_queue_size: int = Field(default=64, ge=1, le=256, description="Bounded queue; 20 concurrent users safe")
    max_parallel_gpu_jobs: int = Field(default=3, ge=1, le=8)
    request_timeout_seconds: float = Field(default=60.0, ge=10.0, le=300.0)
    inference_timeout_seconds: float = Field(default=55.0, ge=10.0, le=120.0)

    default_resolution: int = Field(default=768, ge=512, le=1024)
    max_resolution: int = Field(default=1024, ge=512, le=1024)
    max_image_bytes: int = Field(default=20 * 1024 * 1024, ge=1024 * 1024)
    min_side: int = Field(default=256, ge=64)
    max_side: int = Field(default=1024, ge=512)

    lora_dir: Path = Field(default_factory=lambda: _backend_dir() / "models" / "loras")
    lora_default_scale: float = Field(default=0.85, ge=0.0, le=2.0)

    @field_validator("lora_dir", mode="before")
    @classmethod
    def _lora_dir_path(cls, v: str | Path) -> Path:
        if isinstance(v, str):
            return Path(v)
        return v


_settings: Settings | None = None


def get_settings() -> Settings:
    global _settings
    if _settings is None:
        _settings = Settings()
    return _settings
