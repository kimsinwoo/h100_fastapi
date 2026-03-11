"""ComfyUI API request/response schemas."""

from __future__ import annotations

from typing import Any

from pydantic import BaseModel, Field


class ComfyUIRunRequest(BaseModel):
    """POST /api/comfyui/run body."""
    workflow: dict[str, Any] = Field(..., description="ComfyUI workflow JSON (prompt)")
    pipeline_name: str | None = Field(None, description="pipelines/ 내 파일명(.json 생략 가능). 지정 시 workflow 대신 로드")
    client_id: str | None = Field(None, description="ComfyUI client_id")
    save_to_generated: bool = Field(True, description="True면 static/generated에 저장 후 URL 반환")


class ComfyUIRunResponse(BaseModel):
    """POST /api/comfyui/run response."""
    success: bool = True
    prompt_id: str | None = None
    image_url: str | None = Field(None, description="/static/generated/xxx.png")
    error: str | None = None
