"""
vLLM server configuration. All settings from environment.
"""

from __future__ import annotations

from functools import lru_cache
from typing import Literal

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class VLLMServerSettings(BaseSettings):
    """Production vLLM gateway/embedded server settings."""

    model_config = SettingsConfigDict(
        env_prefix="VLLM_",
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
        protected_namespaces=("settings_",),
    )

    # Backend: use external vLLM OpenAI server (recommended) or embedded engine.
    # Gateway가 7001에서 떠 있을 때 기본값 7001이면 자기 자신을 가리키므로, 반드시 다른 포트 또는 VLLM_USE_MOCK=1 설정.
    backend_url: str = Field(
        default="",
        description="vLLM OpenAI 호환 서버 URL. 비우면 embedded. 게이트웨이 7001 사용 시 여기엔 vLLM이 도는 다른 포트(예 8000) 지정.",
    )
    use_embedded_engine: bool = Field(
        default=False,
        description="If True and backend_url empty, run AsyncLLM in-process.",
    )
    use_mock: bool = Field(
        default=False,
        description="If True, return mock completions without calling vLLM (for testing without GPU/vLLM).",
    )

    # Model (for embedded engine / vLLM serve)
    model_id: str = Field(
        default="Qwen/Qwen3.5-35B-A3B",
        description="Hugging Face model ID. 기본: Qwen3.5-35B-A3B (VLM, 35B params, 3B active).",
    )

    # Concurrency and timeouts
    max_concurrent_requests: int = Field(
        default=100,
        ge=1,
        le=1000,
        description="Max in-flight requests (semaphore).",
    )
    request_timeout_seconds: float = Field(
        default=120.0,
        ge=10.0,
        le=600.0,
        description="Per-request timeout for inference.",
    )
    queue_wait_timeout_seconds: float = Field(
        default=60.0,
        ge=5.0,
        le=300.0,
        description="Max wait time for a semaphore slot before 503.",
    )

    # Server (vLLM gateway only; main app uses app.core.config port 7000)
    host: str = Field(default="0.0.0.0")
    port: int = Field(default=7001, ge=1, le=65535)
    log_level: Literal["DEBUG", "INFO", "WARNING", "ERROR"] = Field(default="INFO")

    # Embedded engine (H100-optimized)
    gpu_memory_utilization: float = Field(default=0.90, ge=0.1, le=1.0)
    max_model_len: int | None = Field(default=None, ge=128, le=131072)
    max_num_seqs: int = Field(default=256, ge=1, le=512, description="Max batch size for continuous batching.")
    tensor_parallel_size: int = Field(default=1, ge=1, le=8)
    dtype: Literal["auto", "float16", "bfloat16"] = Field(default="bfloat16")
    enforce_eager: bool = Field(default=False, description="Set True to avoid CUDA graph for debugging.")
    trust_remote_code: bool = Field(default=True)
    hf_token: str = Field(default="", description="Hugging Face token for gated models.")


@lru_cache
def get_vllm_settings() -> VLLMServerSettings:
    return VLLMServerSettings()
