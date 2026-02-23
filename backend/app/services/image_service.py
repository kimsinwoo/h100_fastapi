"""
Production-grade Z-Image-Turbo img2img service
Optimized for quality, stability, and commercial deployment
"""

from __future__ import annotations

# Z-Image 파이프라인 import 시 JITCallable._set_src() 오류 방지 (PyTorch JIT/compile 호환성)
import os
os.environ.setdefault("PYTORCH_JIT", "0")
os.environ.setdefault("TORCH_COMPILE_DISABLE", "1")

import asyncio
import io
import logging
import time
import warnings
from typing import Any

from app.core.config import get_settings
from app.models.image_prompt_expert import ImagePromptExpert

logger = logging.getLogger(__name__)

_pipeline: Any = None
_lock = asyncio.Lock()
_device: Any = None

# ===== Z-Image-Turbo 권장값 =====
# guidance_scale=0 이면 negative_prompt는 무시됨(공식 문서). 픽셀아트만 1.8로 올려 네거티브 적용.
DEFAULT_GUIDANCE_SCALE = 0.0
PIXEL_ART_GUIDANCE_SCALE = 1.8  # 픽셀아트: voxel/3D 블록 차단하려면 1 이상 필요
DEFAULT_NUM_INFERENCE_STEPS = 8
MODEL_RESOLUTION = 1024

# 스타일별 strength: 픽셀아트는 낮춰야 3D 블록/복셀 방지, 나머지는 각 특성 유지
STRENGTH_BY_STYLE: dict[str, tuple[float, float]] = {
    "pixel art": (0.36, 0.46),      # 매우 낮게 유지해야 순수 2D 스프라이트만 나옴
    "anime": (0.48, 0.56),
    "realistic": (0.46, 0.56),
    "watercolor": (0.48, 0.56),
    "cyberpunk": (0.48, 0.56),
    "oil painting": (0.48, 0.56),
    "sketch": (0.48, 0.56),
    "cinematic": (0.46, 0.54),
    "fantasy art": (0.48, 0.56),
    "3d render": (0.50, 0.58),
}
DEFAULT_STRENGTH_FALLBACK = 0.50
STRENGTH_GLOBAL_MAX = 0.58

# 픽셀 아트 선택 시 네거티브에 추가로 넣어 3D/복셀 완전 차단
PIXEL_ART_NEGATIVE_SUFFIX = (
    ", voxel art, 3D pixel art, blocky 3D, Minecraft style, lego style, "
    "sweater made of blocks, dog made of cubes, volumetric blocks, 2.5D"
)

# 순수 2D 픽셀 아트만 (마인크래프트/복셀/3D 블록 완전 배제)
# ============================================================
# Device
# ============================================================

def _resolve_device():
    import torch
    if torch.cuda.is_available():
        return torch.device("cuda")
    if getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


# ============================================================
# Load Pipeline
# ============================================================

def _load_pipeline_sync():
    global _pipeline, _device

    import os
    # JITCallable._set_src() 호환성 오류 방지: diffusers Z-Image 로드 전에 JIT/compile 비활성화
    os.environ.setdefault("PYTORCH_JIT", "0")
    os.environ.setdefault("TORCH_COMPILE_DISABLE", "1")

    import torch
    # torch.compile 사용 시 일부 환경에서 JIT 오류 발생 가능 → 비활성화
    if getattr(torch, "_dynamo", None) and getattr(torch._dynamo, "config", None):
        try:
            torch._dynamo.config.suppress_errors = True
        except Exception:
            pass

    # xformers + Triton 3.x 호환 오류(JITCallable._set_src) 회피: xformers 미사용으로 로드
    try:
        import diffusers.utils.import_utils as diffusers_import_utils
        diffusers_import_utils.is_xformers_available = lambda: False
    except Exception:
        pass

    try:
        from diffusers import ZImageImg2ImgPipeline
    except ImportError as e:
        err_msg = str(e)
        if "JITCallable" in err_msg or "_set_src" in err_msg:
            logger.exception("Z-Image pipeline import failed (JIT/torch compatibility): %s", e)
            raise RuntimeError(
                "Z-Image 파이프라인 로드 중 JIT 호환 오류가 발생했습니다. "
                "서버 시작 전에 다음 환경 변수를 설정한 뒤 다시 시도하세요: "
                "PYTORCH_JIT=0 TORCH_COMPILE_DISABLE=1"
            ) from e
        logger.error(
            "ZImageImg2ImgPipeline not found. Z-Image-Turbo requires diffusers from git: "
            "pip install git+https://github.com/huggingface/diffusers.git -U (error: %s)",
            e,
        )
        raise RuntimeError(
            "Z-Image-Turbo 파이프라인을 불러올 수 없습니다. "
            "diffusers 최신 버전이 필요합니다: pip install git+https://github.com/huggingface/diffusers.git -U"
        ) from e
    except Exception as e:
        err_msg = str(e)
        if "JITCallable" in err_msg or "_set_src" in err_msg:
            logger.exception("Z-Image pipeline import failed (JIT/torch compatibility): %s", e)
            raise RuntimeError(
                "Z-Image 파이프라인 로드 중 JIT 호환 오류가 발생했습니다. "
                "서버 시작 전에 다음 환경 변수를 설정한 뒤 다시 시도하세요: "
                "PYTORCH_JIT=0 TORCH_COMPILE_DISABLE=1"
            ) from e
        raise

    settings = get_settings()
    _device = _resolve_device()

    dtype = torch.bfloat16 if (_device.type == "cuda" and getattr(torch, "bfloat16", None)) else torch.float32

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")

        try:
            pipe = ZImageImg2ImgPipeline.from_pretrained(
                settings.model_id,
                torch_dtype=dtype,
                low_cpu_mem_usage=True,
            )
        except Exception as e:
            logger.exception("Failed to load model %s: %s", settings.model_id, e)
            raise RuntimeError(
                f"모델 로드 실패: {settings.model_id}. "
                "인터넷 연결·Hugging Face 접근·디스크 공간을 확인하세요. "
                "diffusers는 pip install git+https://github.com/huggingface/diffusers.git 로 설치 권장."
            ) from e

        for method_name in ("enable_attention_slicing", "enable_vae_slicing", "enable_vae_tiling"):
            method = getattr(pipe, method_name, None)
            if callable(method):
                try:
                    method()
                except Exception:
                    pass

        pipe = pipe.to(_device)

    logger.info(
        "Pipeline loaded on %s (dtype=%s)",
        _device,
        dtype,
    )

    _pipeline = pipe
    return pipe


async def get_pipeline():
    global _pipeline
    async with _lock:
        if _pipeline is None:
            loop = asyncio.get_event_loop()
            _pipeline = await loop.run_in_executor(
                None, _load_pipeline_sync
            )
        return _pipeline


# ============================================================
# Inference
# ============================================================

def _resize_keep_ratio(in_w: int, in_h: int, max_side: int) -> tuple[int, int]:
    """입력 비율 유지하며 긴 변이 max_side 이하, 8의 배수로 (out_w, out_h) 계산."""
    if in_w <= 0 or in_h <= 0:
        return (max_side, max_side)
    scale = max_side / max(in_w, in_h)
    out_w = max(64, min(max_side, round(in_w * scale)))
    out_h = max(64, min(max_side, round(in_h * scale)))
    out_w = (out_w // 8) * 8
    out_h = (out_h // 8) * 8
    return (max(64, out_w), max(64, out_h))


def _run_inference_sync(
    image_bytes: bytes,
    prompt: str,
    negative_prompt: str,
    strength: float,
    num_steps: int,
    guidance_scale: float,
    width: int,
    height: int,
    seed: int | None,
) -> bytes:

    import torch
    from PIL import Image

    global _pipeline, _device

    if _pipeline is None:
        raise RuntimeError("Pipeline not loaded")

    img = Image.open(io.BytesIO(image_bytes)).convert("RGB")
    target_w, target_h = width, height
    if img.width != target_w or img.height != target_h:
        img = img.resize((target_w, target_h), Image.Resampling.LANCZOS)

    # Deterministic seed
    generator = torch.Generator(device=_device)
    if seed is not None:
        generator.manual_seed(seed)

    logger.info(
        "Running img2img | steps=%d | guidance=%.2f | strength=%.2f",
        num_steps,
        guidance_scale,
        strength,
    )

    with torch.inference_mode():
        result = _pipeline(
            prompt=prompt,
            negative_prompt=negative_prompt,
            image=img,
            strength=strength,
            num_inference_steps=num_steps,
            guidance_scale=guidance_scale,
            generator=generator,
            output_type="pil",   # 🔥 Let diffusers handle decoding
        )

    image = result.images[0]

    buf = io.BytesIO()
    image.save(buf, format="PNG")
    return buf.getvalue()


# ============================================================
# Public API
# ============================================================

async def run_image_to_image(
    image_bytes: bytes,
    style_key: str,
    custom_prompt: str | None = None,
    strength: float | None = None,
    num_steps: int | None = None,
    size: int | None = None,
    seed: int | None = None,
):

    pipe = await get_pipeline()
    if pipe is None:
        raise RuntimeError("Model not available")

    settings = get_settings()
    style_lower = style_key.lower().strip()

    # ImagePromptExpert + 구성 유지 (복잡한 사진도 레이아웃 유지)
    compiled = ImagePromptExpert.compile(
        style_key, custom_prompt or "", aspect_ratio="1:1"
    )
    prompt = compiled["final_prompt"]
    prompt += (
        ", high detail, sharp focus, preserve fine details and texture, "
        "preserve original composition, same layout and pose, keep subject arrangement, "
        "same subject(s) as reference image, same number of figures or animals, do not change to human or one character"
    )
    negative_prompt = compiled["negative_prompt"]
    if "pixel" in style_lower:
        negative_prompt = negative_prompt + PIXEL_ART_NEGATIVE_SUFFIX

    # 스타일별 strength 상한·기본값
    default_st, max_st = STRENGTH_BY_STYLE.get(
        style_lower, (DEFAULT_STRENGTH_FALLBACK, STRENGTH_GLOBAL_MAX)
    )
    strength = strength if strength is not None else default_st
    strength = max(0.0, min(STRENGTH_GLOBAL_MAX, min(1.0, strength), max_st))

    num_steps = max(1, min(50, num_steps or DEFAULT_NUM_INFERENCE_STEPS))
    guidance_scale = PIXEL_ART_GUIDANCE_SCALE if "pixel" in style_lower else DEFAULT_GUIDANCE_SCALE

    max_side = size or MODEL_RESOLUTION
    from PIL import Image
    with Image.open(io.BytesIO(image_bytes)) as tmp:
        tmp.load()
        in_w, in_h = tmp.width, tmp.height
    target_w, target_h = _resize_keep_ratio(in_w, in_h, max_side)

    loop = asyncio.get_event_loop()
    start = time.perf_counter()

    result = await loop.run_in_executor(
        None,
        lambda: _run_inference_sync(
            image_bytes,
            prompt,
            negative_prompt,
            strength,
            num_steps,
            guidance_scale,
            target_w,
            target_h,
            seed,
        ),
    )

    elapsed = time.perf_counter() - start
    return result, elapsed


# ============================================================
# Utilities
# ============================================================

def is_pipeline_loaded() -> bool:
    return _pipeline is not None


def is_gpu_available() -> bool:
    try:
        import torch
        return torch.cuda.is_available()
    except Exception:
        return False
