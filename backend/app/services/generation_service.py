"""
Async generation service with GPU semaphore. No global blocking.
"""

from __future__ import annotations

import asyncio
import io
import logging
import time
from pathlib import Path
from typing import Optional

from PIL import Image

from app.core.config import get_settings
from app.pipelines.unified import generate_image

logger = logging.getLogger(__name__)

_semaphore: Optional[asyncio.Semaphore] = None


def _get_semaphore() -> asyncio.Semaphore:
    global _semaphore
    if _semaphore is None:
        s = get_settings()
        _semaphore = asyncio.Semaphore(s.gpu_semaphore_limit)
    return _semaphore


async def run_generate(
    prompt: str,
    negative_prompt: str,
    style: str,
    image_bytes: Optional[bytes],
    strength: float,
    steps: int,
    cfg: float,
    seed: Optional[int],
    width: int,
    height: int,
    lora_path: Optional[str],
    lora_scale: float,
) -> tuple[bytes, float]:
    s = get_settings()
    steps = max(s.min_inference_steps, min(s.max_inference_steps, steps))
    cfg = max(1.0, min(20.0, cfg))
    strength = max(0.0, min(1.0, strength))

    pil_image: Optional[Image.Image] = None
    if image_bytes:
        pil_image = Image.open(io.BytesIO(image_bytes)).convert("RGB")

    lora_resolved: Optional[Path] = None
    if lora_path:
        p = Path(lora_path)
        if not p.is_absolute():
            p = s.lora_dir / p
        if p.with_suffix(".safetensors").exists():
            lora_resolved = p.with_suffix(".safetensors")
        elif p.exists():
            lora_resolved = p

    sem = _get_semaphore()
    async with sem:
        loop = asyncio.get_event_loop()
        start = time.perf_counter()
        out_bytes = await loop.run_in_executor(
            None,
            lambda: generate_image(
                prompt=prompt,
                negative_prompt=negative_prompt,
                style=style,
                image=pil_image,
                strength=strength,
                steps=steps,
                cfg=cfg,
                seed=seed,
                width=width,
                height=height,
                lora_path=lora_resolved,
                lora_scale=lora_scale,
            ),
        )
        elapsed = time.perf_counter() - start
    return out_bytes, elapsed
