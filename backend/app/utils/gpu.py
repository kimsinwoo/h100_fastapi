"""
GPU optimization: TF32, memory limits, inference_mode. Call at startup.
"""

from __future__ import annotations

import logging

from app.core.config import get_settings

logger = logging.getLogger(__name__)


def init_gpu() -> None:
    s = get_settings()
    try:
        import torch
        if s.enable_tf32 and torch.cuda.is_available():
            torch.backends.cuda.matmul.allow_tf32 = True
            torch.backends.cudnn.allow_tf32 = True
        logger.info("GPU init: tf32=%s", s.enable_tf32)
    except Exception as e:
        logger.warning("GPU init: %s", e)
