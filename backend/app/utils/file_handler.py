"""
Non-blocking file operations for uploads and generated images.
"""

from __future__ import annotations

import uuid
from pathlib import Path
from typing import BinaryIO

from app.core.config import get_settings


def ensure_generated_dir() -> Path:
    """Ensure static/generated exists; return path."""
    settings = get_settings()
    path = settings.generated_dir
    path.mkdir(parents=True, exist_ok=True)
    return path


def save_upload(content: bytes, suffix: str = ".png") -> str:
    """
    Save uploaded bytes to static/generated. Return filename only (for URL path).
    Uses UUID to avoid collisions.
    """
    ensure_generated_dir()
    name = f"{uuid.uuid4().hex}{suffix}"
    path = get_settings().generated_dir / name
    path.write_bytes(content)
    return name


async def save_upload_async(content: bytes, suffix: str = ".png") -> str:
    """Async wrapper: run save in executor to avoid blocking."""
    import asyncio
    loop = asyncio.get_event_loop()
    return await loop.run_in_executor(None, save_upload, content, suffix)


def read_image_bytes(path: Path) -> bytes:
    return path.read_bytes()


def get_generated_url(filename: str) -> str:
    """Return URL path for a file under static/generated (e.g. /static/generated/abc.png)."""
    settings = get_settings()
    return f"/static/{settings.generated_dir_name}/{filename}"
