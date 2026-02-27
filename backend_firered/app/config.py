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

    # Inference defaults (H100 target: sub-2s at 768, 8–12 steps)
    DEFAULT_STEPS: int = Field(default=12, ge=1, le=100)
    DEFAULT_GUIDANCE: float = Field(default=6.5, ge=1.0, le=20.0)
    DEFAULT_STRENGTH: float = Field(default=0.65, ge=0.0, le=1.0)
    # Resolution: if any side > MAX_RESOLUTION_INPUT, downscale to max MAX_RESOLUTION (keep aspect)
    MAX_RESOLUTION: int = Field(default=768, ge=512, le=1024)
    MAX_RESOLUTION_INPUT: int = Field(default=1024, ge=512, le=2048)
    # Production: cap steps to avoid slow requests
    PRODUCTION_STEPS_CAP: int = Field(default=20, ge=8, le=50)
    # If guidance_scale <= 0 and not DEBUG, override to DEFAULT_GUIDANCE
    DEBUG_MODE: bool = Field(default=False)

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
