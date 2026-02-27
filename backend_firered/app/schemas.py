"""
Request/response and error schemas for the edit API.
"""

from __future__ import annotations

from typing import Any

from pydantic import BaseModel, Field


class ErrorResponse(BaseModel):
    """Structured error payload for 4xx/5xx."""

    error: str = Field(..., description="Short error code or type")
    detail: str = Field(..., description="Human-readable detail")
    request_id: str = Field(..., description="Request correlation ID")


def error_payload(error: str, detail: str, request_id: str) -> dict[str, Any]:
    """Build JSON-serializable error body."""
    return ErrorResponse(error=error, detail=detail, request_id=request_id).model_dump()
