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
    upload_max_size_mb: int = Field(default=20, ge=1, le=100)

    # Z-Image model
    model_id: str = Field(default="Tongyi-MAI/Z-Image-Turbo")
    device_preference: Literal["cuda", "cpu", "auto"] = Field(default="auto")
    default_strength: float = Field(default=0.6, ge=0.0, le=1.0)
    default_steps: int = Field(default=9, ge=1, le=50)
    default_size: int = Field(default=1024, ge=512, le=2048)
    use_autocast: bool = Field(default=True)

    # CORS
    cors_origins: str = Field(default="*")
    cors_allow_credentials: bool = Field(default=True)

    # Rate limiting
    rate_limit_requests: int = Field(default=30, ge=1, le=200)
    rate_limit_window_seconds: int = Field(default=60, ge=1, le=3600)

    @property
    def generated_dir(self) -> Path:
        return self.static_dir / self.generated_dir_name

    @property
    def upload_max_bytes(self) -> int:
        return self.upload_max_size_mb * 1024 * 1024


@lru_cache
def get_settings() -> Settings:
    return Settings()
