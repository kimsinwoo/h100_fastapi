"""
SD 1.5 config. Paths relative to zimage_webapp/backend when run from there.
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
        env_prefix="SD15_",
    )

    # Hugging Face에서 다운로드 후 로컬에서 추론 (로컬 경로 없으면 model_id 사용)
    model_id: str = Field(
        default="runwayml/stable-diffusion-v1-5",
        description="Hugging Face model ID when not using local model_path",
    )
    model_path: Path | None = Field(
        default=None,
        description="Local path to SD 1.5 base model; set to use local instead of downloading from HF",
    )

    @field_validator("model_path", mode="before")
    @classmethod
    def _model_path_path(cls, v: str | Path | None) -> Path | None:
        if v is None or v == "":
            return None
        return Path(v) if isinstance(v, str) else v

    lora_dir: Path = Field(
        default_factory=lambda: _backend_dir() / "models" / "loras",
        description="Directory for LoRA adapters",
    )

    @field_validator("lora_dir", mode="before")
    @classmethod
    def _lora_dir_path(cls, v: str | Path) -> Path:
        if isinstance(v, str):
            return Path(v)
        return v

    device: str = Field(default="cuda")
    torch_dtype: str = Field(default="float16")

    default_strength: float = Field(default=0.7, ge=0.0, le=1.0)
    default_steps: int = Field(default=30, ge=1, le=100)
    default_cfg: float = Field(default=8.0, ge=1.0, le=20.0)
    lora_weight: float = Field(default=1.0, ge=0.0, le=2.0)

    strength_min: float = Field(default=0.6, ge=0.0, le=1.0)
    strength_max: float = Field(default=0.8, ge=0.0, le=1.0)
    cfg_min: float = Field(default=7.0, ge=1.0, le=20.0)
    cfg_max: float = Field(default=9.0, ge=1.0, le=20.0)
    steps_min: int = Field(default=25, ge=1, le=100)
    steps_max: int = Field(default=40, ge=1, le=100)

    max_resolution: int = Field(default=1024, ge=512, le=1024)
    queue_max_size: int = Field(default=64, ge=1, le=256)
    worker_timeout_seconds: float = Field(default=120.0, ge=10.0)

    enable_upscale: bool = Field(default=False)
    upscale_model_path: str | None = Field(default=None)


def get_settings() -> Settings:
    return Settings()
