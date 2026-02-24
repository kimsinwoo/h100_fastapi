"""Validation and image utils: base64, dimensions, no Any."""
from __future__ import annotations

import base64
import io
import re
from typing import Tuple

from PIL import Image

from app.sdxl_prod.config import get_settings

SUBJECT_PLACEHOLDER = "{subject}"
MAX_IMAGE_BYTES_DEFAULT = 20 * 1024 * 1024
MIN_SIDE_DEFAULT = 256
MAX_SIDE_DEFAULT = 1024
ALIGN = 8


def validate_base64_input(b64: str | None, max_bytes: int | None = None) -> bytes:
    """Validate base64 string and decode. Raises ValueError on invalid or oversized."""
    if b64 is None or not isinstance(b64, str) or not b64.strip():
        raise ValueError("image_base64 must be a non-empty string")
    b64_clean = b64.strip()
    if not re.match(r"^[A-Za-z0-9+/]*=*$", b64_clean):
        raise ValueError("Invalid base64 characters")
    try:
        raw = base64.b64decode(b64_clean, validate=True)
    except Exception as e:
        raise ValueError(f"Invalid base64: {e}") from e
    limit = max_bytes or get_settings().max_image_bytes
    if len(raw) > limit or len(raw) == 0:
        raise ValueError(f"Image size out of range (max {limit} bytes)")
    return raw


def encode_bytes_to_base64(data: bytes) -> str:
    return base64.b64encode(data).decode("ascii")


def _align8(x: int, min_side: int) -> int:
    return max(min_side, (x // ALIGN) * ALIGN)


def decode_image_and_validate(
    image_bytes: bytes,
    max_side: int | None = None,
    min_side: int | None = None,
) -> Tuple[Image.Image, int, int]:
    """Decode image to RGB PIL; resize if over max_side; align to 8. Returns (img, w, h)."""
    s = get_settings()
    max_s = max_side if max_side is not None else s.max_resolution
    min_s = min_side if min_side is not None else s.min_side
    try:
        img = Image.open(io.BytesIO(image_bytes)).convert("RGB")
    except Exception as e:
        raise ValueError(f"Invalid image: {e}") from e
    w, h = img.size
    if w < min_s or h < min_s:
        raise ValueError(f"Image too small (min {min_s}px per side)")
    if w > max_s or h > max_s:
        scale = max_s / max(w, h)
        new_w = _align8(int(round(w * scale)), min_s)
        new_h = _align8(int(round(h * scale)), min_s)
        img = img.resize((new_w, new_h), Image.Resampling.LANCZOS)
        w, h = img.size
    return img, w, h


def image_to_bytes_png(img: Image.Image) -> bytes:
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    return buf.getvalue()


def build_positive_prompt(template: str, user_prompt: str) -> str:
    subject = user_prompt.strip() if user_prompt else "subject"
    if SUBJECT_PLACEHOLDER not in template:
        return f"{template}, {subject}"
    return template.replace(SUBJECT_PLACEHOLDER, subject)
