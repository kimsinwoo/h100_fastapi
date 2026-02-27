"""
Application configuration. Pydantic v2 BaseSettings.
HunyuanImage-3.0-Instruct backend only.
"""

from __future__ import annotations

from functools import lru_cache
from pathlib import Path

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """Environment-based settings. All keys overridable via env vars."""

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
        env_ignore_empty=True,
        protected_namespaces=("settings_",),
    )

    # Model path (no dots in dir name — use e.g. ./HunyuanImage-3-Instruct)
    MODEL_ID: str = Field(
        default="./HunyuanImage-3-Instruct",
        description="Path to HunyuanImage-3.0-Instruct (downloaded without dots in dir name)",
    )

    # Concurrency & timeouts
    MAX_CONCURRENT_JOBS: int = Field(default=1, ge=1, le=4)
    TIMEOUT_SECONDS: int = Field(default=120, ge=30, le=300)

    # Inference defaults (HunyuanImage-3.0-Instruct)
    DEFAULT_STEPS: int = Field(default=28, ge=8, le=50)
    DEFAULT_BOT_TASK: str = Field(default="think_recaption")
    DEFAULT_USE_SYSTEM_PROMPT: str = Field(default="en_unified")
    DEFAULT_IMAGE_SIZE: str = Field(default="auto")

    # Server
    host: str = Field(default="0.0.0.0")
    port: int = Field(default=8000, ge=1, le=65535)

    def model_path(self) -> Path:
        """Resolved model path for loading."""
        return Path(self.MODEL_ID).expanduser().resolve()


@lru_cache
def get_settings() -> Settings:
    """Cached settings singleton."""
    return Settings()
