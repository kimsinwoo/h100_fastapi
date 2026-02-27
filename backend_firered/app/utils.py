"""
Shared utilities: image loading, request IDs, GPU memory.
"""

from __future__ import annotations

import io
import logging
import uuid

from fastapi import UploadFile
from PIL import Image

logger = logging.getLogger(__name__)


def generate_request_id() -> str:
    """Return a short unique request ID for logging and error responses."""
    return uuid.uuid4().hex[:12]


async def load_image_rgb(upload: UploadFile) -> Image.Image:
    """
    Read upload content and return a PIL Image in RGB.
    Raises ValueError if the file is not a valid image.
    """
    content = await upload.read()
    if not content:
        raise ValueError("Empty file")
    try:
        img = Image.open(io.BytesIO(content)).convert("RGB")
    except Exception as e:
        raise ValueError(f"Invalid image: {e}") from e
    return img


def get_gpu_memory_mb() -> float | None:
    """Return current GPU memory allocated in MB, or None if CUDA unavailable."""
    try:
        import torch
        if not torch.cuda.is_available():
            return None
        return torch.cuda.memory_allocated() / (1024 * 1024)
    except Exception:
        return None
