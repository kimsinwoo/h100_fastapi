"""
Request/response and error schemas for the edit API.
"""

from __future__ import annotations

from typing import Any

from pydantic import BaseModel, Field


class ErrorResponse(BaseModel):
    """Structured error payload for 4xx/5xx."""

    error: str = Field(..., description="Error code or type (e.g. BadRequest)")
    message: str = Field(..., description="Human-readable error message")
    request_id: str = Field(..., description="Request correlation ID")


def error_payload(error: str, message: str, request_id: str) -> dict[str, Any]:
    """Build JSON-serializable error body for API responses."""
    return ErrorResponse(error=error, message=message, request_id=request_id).model_dump()
