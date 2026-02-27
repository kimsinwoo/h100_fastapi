"""
Shared utilities: image loading, request IDs, GPU memory, resolution normalization.
"""

from __future__ import annotations

import io
import logging
import uuid
from typing import Any

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


def normalize_resolution(
    image: Image.Image,
    max_side: int = 768,
    trigger_threshold: int = 1024,
    resample: int = Image.Resampling.LANCZOS,
) -> tuple[Image.Image, tuple[int, int], tuple[int, int]]:
    """
    If any side > trigger_threshold, downscale so the longest side is max_side.
    Keeps aspect ratio. Uses Lanczos. Returns (image, original_size, new_size).
    """
    w, h = image.size
    original_size = (w, h)
    if w <= trigger_threshold and h <= trigger_threshold:
        return image, original_size, original_size
    scale = max_side / max(w, h)
    new_w = int(round(w * scale))
    new_h = int(round(h * scale))
    new_w = max(new_w, 8)
    new_h = max(new_h, 8)
    new_size = (new_w, new_h)
    resized = image.resize(new_size, resample=resample)
    logger.info(
        "resolution_normalized original=%s resized=%s max_side=%s",
        original_size,
        new_size,
        max_side,
    )
    return resized, original_size, new_size


def get_gpu_info() -> dict[str, Any] | None:
    """
    Return GPU info for /health/gpu. Call outside inference hot path.
    Includes device name, allocated/reserved memory, compute capability.
    """
    try:
        import torch
        if not torch.cuda.is_available():
            return None
        name = torch.cuda.get_device_name(0)
        allocated = torch.cuda.memory_allocated(0) / (1024 * 1024)
        reserved = torch.cuda.memory_reserved(0) / (1024 * 1024)
        cap = torch.cuda.get_device_capability(0)
        return {
            "device_name": name,
            "allocated_mb": round(allocated, 2),
            "reserved_mb": round(reserved, 2),
            "compute_capability": f"{cap[0]}.{cap[1]}",
        }
    except Exception:
        return None
