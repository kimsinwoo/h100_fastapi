"""
LTX-2 Image-to-Video (Lightricks LTX-2).
사진 + 프롬프트 → 동영상 생성. Hugging Face diffusers LTX2ImageToVideoPipeline 사용.
Ref: https://huggingface.co/docs/diffusers/main/en/api/pipelines/ltx2
"""

from __future__ import annotations

import asyncio
import io
import logging
import time
from typing import Any

from app.core.config import get_settings

logger = logging.getLogger(__name__)

_pipeline: Any = None
_pipeline_lock = asyncio.Lock()

# 기본 해상도/프레임 (VRAM에 맞게 조정 가능)
DEFAULT_WIDTH = 768
DEFAULT_HEIGHT = 512
DEFAULT_NUM_FRAMES = 81  # 121은 무거우므로 81로 줄임 (약 3.4초 @ 24fps)
DEFAULT_FRAME_RATE = 24.0
DEFAULT_NUM_STEPS = 25
DEFAULT_GUIDANCE_SCALE = 4.0
DEFAULT_NEGATIVE = (
    "worst quality, inconsistent motion, blurry, jittery, distorted, "
    "shaky, glitchy, deformed, motion smear, motion artifacts, bad anatomy, static."
)


def _load_pipeline_sync() -> Any:
    import torch
    try:
        from diffusers import LTX2ImageToVideoPipeline
    except ImportError:
        try:
            from diffusers.pipelines.ltx2 import LTX2ImageToVideoPipeline
        except ImportError as e:
            raise ImportError(
                "LTX2ImageToVideoPipeline을 불러올 수 없습니다. LTX-2 비디오는 diffusers main 브랜치 필요 (0.37 미출시). "
                "설치: pip install git+https://github.com/huggingface/diffusers.git"
            ) from e

    global _pipeline
    model_id = getattr(get_settings(), "ltx2_model_id", "Lightricks/LTX-2")
    dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
    _pipeline = LTX2ImageToVideoPipeline.from_pretrained(model_id, torch_dtype=dtype)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    if device == "cuda":
        _pipeline.enable_sequential_cpu_offload(device=device)
    else:
        _pipeline = _pipeline.to(device)
    logger.info("LTX-2 Image-to-Video pipeline loaded (model=%s)", model_id)
    return _pipeline


async def get_video_pipeline() -> Any:
    global _pipeline
    async with _pipeline_lock:
        if _pipeline is None:
            loop = asyncio.get_event_loop()
            _pipeline = await loop.run_in_executor(None, _load_pipeline_sync)
        return _pipeline


def _run_image_to_video_sync(
    image_bytes: bytes,
    prompt: str,
    negative_prompt: str,
    width: int,
    height: int,
    num_frames: int,
    frame_rate: float,
    num_inference_steps: int,
    guidance_scale: float,
    seed: int | None,
) -> bytes:
    import torch
    from PIL import Image, ImageOps

    global _pipeline
    if _pipeline is None:
        raise RuntimeError("Video pipeline not loaded")

    img = Image.open(io.BytesIO(image_bytes))
    img = ImageOps.exif_transpose(img)
    img = img.convert("RGB")
    # 해상도 맞춤 (비율 유지 후 crop 또는 resize)
    w, h = img.size
    scale_w = width / w
    scale_h = height / h
    scale = max(scale_w, scale_h)
    new_w = max(width, int(w * scale))
    new_h = max(height, int(h * scale))
    img = img.resize((new_w, new_h), Image.Resampling.LANCZOS)
    # center crop
    left = (new_w - width) // 2
    top = (new_h - height) // 2
    img = img.crop((left, top, left + width, top + height))

    generator = None
    if seed is not None:
        generator = torch.Generator(device="cpu").manual_seed(seed)

    logger.info(
        "LTX-2 image2video | %dx%d | frames=%d | steps=%d",
        width, height, num_frames, num_inference_steps,
    )
    out = _pipeline(
        image=img,
        prompt=prompt,
        negative_prompt=negative_prompt or DEFAULT_NEGATIVE,
        width=width,
        height=height,
        num_frames=num_frames,
        frame_rate=frame_rate,
        num_inference_steps=num_inference_steps,
        guidance_scale=guidance_scale,
        output_type="np",
        return_dict=False,
        generator=generator,
    )
    video_frames = out[0]
    audio = out[1] if len(out) > 1 else None

    from diffusers.pipelines.ltx2.export_utils import encode_video
    import tempfile
    import os

    with tempfile.NamedTemporaryFile(suffix=".mp4", delete=False) as f:
        tmp_path = f.name
    try:
        kwargs = {"output_path": tmp_path, "fps": frame_rate}
        if audio is not None and len(audio) > 0 and hasattr(_pipeline, "vocoder") and _pipeline.vocoder is not None:
            kwargs["audio"] = audio[0].float().cpu()
            kwargs["audio_sample_rate"] = _pipeline.vocoder.config.output_sampling_rate
        encode_video(video_frames[0], **kwargs)
        with open(tmp_path, "rb") as f:
            return f.read()
    finally:
        try:
            os.unlink(tmp_path)
        except Exception:
            pass


async def run_image_to_video(
    image_bytes: bytes,
    prompt: str,
    negative_prompt: str | None = None,
    width: int = DEFAULT_WIDTH,
    height: int = DEFAULT_HEIGHT,
    num_frames: int = DEFAULT_NUM_FRAMES,
    frame_rate: float = DEFAULT_FRAME_RATE,
    num_inference_steps: int = DEFAULT_NUM_STEPS,
    guidance_scale: float = DEFAULT_GUIDANCE_SCALE,
    seed: int | None = None,
) -> tuple[bytes, float]:
    await get_video_pipeline()
    loop = asyncio.get_event_loop()
    start = time.perf_counter()
    result = await loop.run_in_executor(
        None,
        lambda: _run_image_to_video_sync(
            image_bytes,
            prompt,
            negative_prompt or DEFAULT_NEGATIVE,
            width,
            height,
            num_frames,
            frame_rate,
            num_inference_steps,
            guidance_scale,
            seed,
        ),
    )
    elapsed = time.perf_counter() - start
    return result, elapsed


def is_video_pipeline_loaded() -> bool:
    return _pipeline is not None
