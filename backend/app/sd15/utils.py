"""Image utils: base64, PIL, resize to multiple of 8, aspect ratio preserved."""
from __future__ import annotations

import base64
import io
from typing import Tuple

from PIL import Image

MAX_IMAGE_BYTES = 20 * 1024 * 1024
MAX_SIDE = 1024
MIN_SIDE = 64
ALIGN = 8


def decode_base64_to_bytes(b64: str) -> bytes:
    if not b64 or not isinstance(b64, str):
        raise ValueError("image_base64 must be a non-empty string")
    try:
        raw = base64.b64decode(b64, validate=True)
    except Exception as e:
        raise ValueError(f"Invalid base64: {e}") from e
    if len(raw) > MAX_IMAGE_BYTES or len(raw) == 0:
        raise ValueError("Invalid image size")
    return raw


def encode_bytes_to_base64(data: bytes) -> str:
    return base64.b64encode(data).decode("ascii")


def _align8(x: int) -> int:
    return max(MIN_SIDE, (x // ALIGN) * ALIGN)


def decode_image_and_validate(
    image_bytes: bytes,
    max_side: int = MAX_SIDE,
    min_side: int = MIN_SIDE,
) -> Tuple[Image.Image, int, int]:
    try:
        img = Image.open(io.BytesIO(image_bytes)).convert("RGB")
    except Exception as e:
        raise ValueError(f"Invalid image: {e}") from e
    w, h = img.size
    if w < min_side or h < min_side:
        raise ValueError(f"Image too small (min {min_side}px per side)")
    if w > max_side or h > max_side:
        img = resize_keep_aspect(img, max_side)
        w, h = img.size
    return img, w, h


def resize_keep_aspect(img: Image.Image, max_side: int) -> Image.Image:
    w, h = img.size
    if w <= max_side and h <= max_side:
        new_w, new_h = _align8(w), _align8(h)
        if (new_w, new_h) != (w, h):
            return img.resize((new_w, new_h), Image.Resampling.LANCZOS)
        return img
    scale = max_side / max(w, h)
    new_w = _align8(int(round(w * scale)))
    new_h = _align8(int(round(h * scale)))
    return img.resize((new_w, new_h), Image.Resampling.LANCZOS)


def image_to_bytes_png(img: Image.Image) -> bytes:
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    return buf.getvalue()
