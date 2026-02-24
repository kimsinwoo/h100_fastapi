"""Optional Real-ESRGAN 2x. On failure return original. No crash."""
from __future__ import annotations

import io
import logging

from app.sd15.config import get_settings

logger = logging.getLogger(__name__)

_upsampler: object | None = None
_loaded = False


def _load_upsampler() -> object | None:
    global _upsampler, _loaded
    if _loaded:
        return _upsampler
    _loaded = True
    s = get_settings()
    if not s.enable_upscale:
        return None
    try:
        from realesrgan import RealESRGANer
        from basicsr.archs.rrdbnet_arch import RRDBNet
        import torch
        model_name = s.upscale_model_path or "RealESRGAN_x2plus"
        model = RRDBNet(num_in_ch=3, num_out_ch=3, num_feat=64, num_block=23, num_grow_ch=32, scale=2)
        _upsampler = RealESRGANer(
            scale=2, model_path=model_name, model=model,
            device="cuda" if torch.cuda.is_available() else "cpu",
        )
        logger.info("Real-ESRGAN 2x loaded")
        return _upsampler
    except Exception as e:
        logger.warning("Real-ESRGAN not available: %s", e)
        _upsampler = None
        return None


def upscale_image_if_enabled(image_bytes: bytes, upscale_requested: bool) -> bytes:
    if not upscale_requested:
        return image_bytes
    up = _load_upsampler()
    if up is None:
        return image_bytes
    try:
        from PIL import Image
        import numpy as np
        img = Image.open(io.BytesIO(image_bytes)).convert("RGB")
        out, _ = up.enhance(np.array(img), outscale=2)
        buf = io.BytesIO()
        Image.fromarray(out).save(buf, format="PNG")
        return buf.getvalue()
    except Exception as e:
        logger.warning("Upscale failed: %s", e)
        return image_bytes
