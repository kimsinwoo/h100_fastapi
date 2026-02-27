"""
Legacy module: style list only. No prompt injection.
All prompt building is in app.utils.prompt_builder.
"""

from __future__ import annotations

from app.utils.prompt_builder import get_allowed_style_keys as _get_allowed_style_keys


class ImagePromptExpert:
    """Thin wrapper for API compatibility. Do not use for prompt building."""

    DEFAULT_STYLE = "realistic"

    @classmethod
    def get_allowed_style_keys(cls) -> list[str]:
        """API 검증·스타일 목록용. prompt_builder와 동기화."""
        return _get_allowed_style_keys()
