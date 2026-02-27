"""
Application configuration. Pydantic v2 BaseSettings.
FireRed-Image-Edit-1.0 backend only.
"""

from __future__ import annotations

from functools import lru_cache
from typing import Literal

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """Environment-based settings. All keys are overridable via env vars."""

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
        env_ignore_empty=True,
        protected_namespaces=("settings_",),
    )

    # Model
    MODEL_ID: str = Field(
        default="FireRedTeam/FireRed-Image-Edit-1.0",
        description="HuggingFace model ID for FireRed image editing",
    )

    # Concurrency & timeouts
    MAX_CONCURRENT_JOBS: int = Field(default=2, ge=1, le=16)
    TIMEOUT_SECONDS: int = Field(default=120, ge=10, le=300)

    # Inference defaults
    DEFAULT_STEPS: int = Field(default=28, ge=1, le=100)
    DEFAULT_GUIDANCE: float = Field(default=7.0, ge=1.0, le=20.0)

    # Server (uvicorn 기존 사용 방식과 동일)
    host: str = Field(default="0.0.0.0")
    port: int = Field(default=7000, ge=1, le=65535)

    # Optional: future distilled / LoRA
    use_safetensors: bool = Field(default=True)
    torch_dtype: Literal["float16", "bfloat16"] = Field(default="float16")


@lru_cache
def get_settings() -> Settings:
    """Cached settings singleton."""
    return Settings()
