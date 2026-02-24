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
import contextlib
import io
import logging
import time
import warnings
from typing import Any

from app.core.config import get_settings
from app.lora.manager import load_lora
from app.models.image_prompt_expert import ImagePromptExpert
from app.models.style_presets import get_style_negative_prompt

# 스타일 키(소문자) -> lora_output 내 파일명. 학습된 LoRA가 있으면 추론 시 로드
STYLE_TO_LORA_FILENAME: dict[str, str] = {
    "3d render": "3d_render.safetensors",
    "cyberpunk": "cyberpunk.safetensors",
    "pixel art": "pixel_art.safetensors",
}
LORA_SCALE = 0.85

logger = logging.getLogger(__name__)

_pipeline: Any = None
_lock = asyncio.Lock()
_device: Any = None
_use_omnigen: bool = False  # True면 OmniGen(Omni) 파이프라인 사용 (H100 등)

# ===== Z-Image-Turbo 권장값 =====
# guidance_scale=0 이면 negative_prompt는 무시됨(공식 문서). 픽셀아트만 1.8로 올려 네거티브 적용.
DEFAULT_GUIDANCE_SCALE = 0.0
PIXEL_ART_GUIDANCE_SCALE = 1.8  # 픽셀아트: voxel/3D 블록 차단하려면 1 이상 필요
DEFAULT_NUM_INFERENCE_STEPS = 8
MODEL_RESOLUTION = 1024

# 스타일별 strength: 픽셀아트는 낮춰야 3D 블록/복셀 방지, 나머지는 각 특성 유지
STRENGTH_BY_STYLE: dict[str, tuple[float, float]] = {
    "pixel art": (0.23, 0.38),      # 매우 낮게 유지해야 순수 2D 스프라이트만 나옴
    "anime": (0.48, 0.56),
    "realistic": (0.46, 0.56),
    "watercolor": (0.48, 0.56),
    "cyberpunk": (0.48, 0.56),
    "oil painting": (0.48, 0.56),
    "sketch": (0.48, 0.56),
    "cinematic": (0.46, 0.54),
    "fantasy art": (0.48, 0.56),
    "3d render": (0.50, 0.58),
    "omni": (0.65, 0.80),           # Omni-Image-Editor: 원본 유지 + 디테일 강화 (0.6~0.8)
}
DEFAULT_STRENGTH_FALLBACK = 0.50
STRENGTH_GLOBAL_MAX = 0.58

# Omni-Image-Editor 참고 (https://huggingface.co/spaces/selfit-camera/Omni-Image-Editor): num_inference_steps=50, guidance_scale=7.5
OMNI_NUM_STEPS = 50                # Omni-Image-Editor pipeline 기본값 50
OMNI_GUIDANCE_SCALE = 7.5           # 7~8 (Omni pipeline 기본 7.5)
OMNI_STEPS_MAX = 70
OMNI_STRENGTH_MAX = 0.80

# 픽셀 아트 선택 시 네거티브에 추가로 넣어 3D/복셀 완전 차단
PIXEL_ART_NEGATIVE_SUFFIX = (
    ", voxel art, 3D pixel art, blocky 3D, Minecraft style, lego style, "
    "sweater made of blocks, dog made of cubes, volumetric blocks, 2.5D, only 2D image and picture"
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
# Load Pipeline (단일 Z-Image-Turbo)
# ============================================================

def _load_pipeline_sync():
    global _pipeline, _device, _use_omnigen

    import os
    os.environ.setdefault("PYTORCH_JIT", "0")
    os.environ.setdefault("TORCH_COMPILE_DISABLE", "1")

    import torch
    if getattr(torch, "_dynamo", None) and getattr(torch._dynamo, "config", None):
        try:
            torch._dynamo.config.suppress_errors = True
        except Exception:
            pass

    settings = get_settings()
    _settings = settings
    if not getattr(_settings, "enable_xformers", False):
        try:
            import diffusers.utils.import_utils as diffusers_import_utils
            diffusers_import_utils.is_xformers_available = lambda: False
        except Exception:
            pass

    _device = _resolve_device()
    dtype = torch.bfloat16 if (_device.type == "cuda" and getattr(torch, "bfloat16", None)) else torch.float32

    # H100 등: OmniGen(Omni) 사용 (use_omnigen=True)
    if getattr(settings, "use_omnigen", False):
        try:
            from diffusers import OmniGenPipeline
        except ImportError as e:
            logger.warning("OmniGenPipeline not found, falling back to Z-Image: %s", e)
        else:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                try:
                    pipe = OmniGenPipeline.from_pretrained(
                        getattr(settings, "omnigen_model_id", "Shitao/OmniGen-v1-diffusers"),
                        torch_dtype=dtype,
                    )
                    pipe = pipe.to(_device)
                    for name in ("enable_attention_slicing", "enable_vae_slicing", "enable_vae_tiling"):
                        fn = getattr(pipe, name, None)
                        if callable(fn):
                            try:
                                fn()
                            except Exception:
                                pass
                    _pipeline = pipe
                    _use_omnigen = True
                    logger.info("OmniGen (Omni) pipeline loaded on %s (dtype=%s)", _device, dtype)
                    return pipe
                except Exception as e:
                    logger.exception("OmniGen load failed: %s", e)
                    raise RuntimeError(
                        f"OmniGen 모델 로드 실패. USE_OMNIGEN=false 로 Z-Image 사용 가능. 원인: {e}"
                    ) from e

    # Z-Image-Turbo
    _use_omnigen = False
    try:
        from diffusers import ZImageImg2ImgPipeline
    except ImportError as e:
        err_msg = str(e)
        if "JITCallable" in err_msg or "_set_src" in err_msg:
            raise RuntimeError(
                "Z-Image 파이프라인 로드 중 JIT 호환 오류. PYTORCH_JIT=0 TORCH_COMPILE_DISABLE=1 설정 후 재시도."
            ) from e
        raise RuntimeError(
            "ZImageImg2ImgPipeline을 찾을 수 없습니다. pip install git+https://github.com/huggingface/diffusers.git -U"
        ) from e

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        pipe = ZImageImg2ImgPipeline.from_pretrained(
            settings.model_id,
            torch_dtype=dtype,
            low_cpu_mem_usage=True,
        )
        for method_name in ("enable_attention_slicing", "enable_vae_slicing", "enable_vae_tiling"):
            method = getattr(pipe, method_name, None)
            if callable(method):
                try:
                    method()
                except Exception:
                    pass
        pipe = pipe.to(_device)

    logger.info("Z-Image-Turbo pipeline loaded on %s (dtype=%s)", _device, dtype)
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


def is_pipeline_loaded() -> bool:
    """이미지 생성 파이프라인 로드 여부 (OmniGen 또는 Z-Image, health 등에서 사용)."""
    return _pipeline is not None


def is_omnigen_in_use() -> bool:
    """현재 OmniGen(Omni) 파이프라인 사용 중인지."""
    return _use_omnigen


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
    from PIL import Image, ImageOps

    global _pipeline, _device

    if _pipeline is None:
        raise RuntimeError("Pipeline not loaded")

    # 전처리: EXIF 방향 적용 + 비율 유지 리사이즈 (Omni-Image-Editor 스타일: LANCZOS, 원본 구조 유지)
    img = Image.open(io.BytesIO(image_bytes))
    img = ImageOps.exif_transpose(img)
    img = img.convert("RGB")
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

    # Omni-Image-Editor 참고: optimized_inference_mode (cudnn benchmark, tf32, flash_sdp) + autocast bfloat16
    @contextlib.contextmanager
    def _optimized_inference_context():
        if _device.type != "cuda":
            yield
            return
        orig_benchmark = torch.backends.cudnn.benchmark
        orig_tf32_matmul = torch.backends.cuda.matmul.allow_tf32
        orig_tf32_cudnn = torch.backends.cudnn.allow_tf32
        try:
            torch.backends.cudnn.benchmark = True
            torch.backends.cuda.matmul.allow_tf32 = True
            torch.backends.cudnn.allow_tf32 = True
            if getattr(torch.backends.cuda, "enable_flash_sdp", None) is not None:
                try:
                    torch.backends.cuda.enable_flash_sdp(True)
                except Exception:
                    pass
            yield
        finally:
            torch.backends.cudnn.benchmark = orig_benchmark
            torch.backends.cuda.matmul.allow_tf32 = orig_tf32_matmul
            torch.backends.cudnn.allow_tf32 = orig_tf32_cudnn

    use_autocast = _device.type == "cuda" and getattr(torch, "bfloat16", None) and torch.cuda.is_bf16_supported()
    ctx_autocast = torch.autocast("cuda", dtype=torch.bfloat16) if use_autocast else contextlib.nullcontext()

    with torch.inference_mode(), _optimized_inference_context(), ctx_autocast:
        result = _pipeline(
            prompt=prompt,
            negative_prompt=negative_prompt,
            image=img,
            strength=strength,
            num_inference_steps=num_steps,
            guidance_scale=guidance_scale,
            generator=generator,
            output_type="pil",
        )

    image = result.images[0]

    buf = io.BytesIO()
    image.save(buf, format="PNG")
    return buf.getvalue()


def _run_inference_omnigen_sync(
    image_bytes: bytes,
    prompt: str,
    seed: int | None,
    num_steps: int = 50,
    guidance_scale: float = 2.0,
    img_guidance_scale: float = 1.6,
) -> bytes:
    """OmniGen(Omni) 이미지 편집: diffusers 요구 형식 <img><|image_1|></img> + input_images."""
    import torch
    from PIL import Image, ImageOps

    global _pipeline, _device

    if _pipeline is None:
        raise RuntimeError("OmniGen pipeline not loaded")
    img = Image.open(io.BytesIO(image_bytes))
    img = ImageOps.exif_transpose(img)
    img = img.convert("RGB")

    generator = torch.Generator(device=_device)
    if seed is not None:
        generator.manual_seed(seed)

    omni_prompt = "<img><|image_1|></img> " + prompt
    # OmniGen VAE decode 시 bfloat16 autocast가 하얀 출력을 유발할 수 있어 비활성화
    use_autocast = False
    ctx = torch.autocast("cuda", dtype=torch.bfloat16) if use_autocast else contextlib.nullcontext()

    logger.info("Running OmniGen img edit | steps=%d | guidance=%.2f | img_guidance=%.2f", num_steps, guidance_scale, img_guidance_scale)
    with torch.inference_mode(), ctx:
        result = _pipeline(
            prompt=omni_prompt,
            input_images=[img],
            guidance_scale=guidance_scale,
            img_guidance_scale=img_guidance_scale,
            num_inference_steps=num_steps,
            use_input_image_size_as_output=True,
            max_input_image_size=1024,
            generator=generator,
            output_type="pil",
        )
    images = result.images if hasattr(result, "images") else result
    if isinstance(images, list):
        out_pil = images[0] if images else None
    else:
        out_pil = images
    if out_pil is None:
        raise RuntimeError("OmniGen returned no image")
    if isinstance(out_pil, Image.Image):
        out_pil = out_pil.convert("RGB")
    else:
        import numpy as np
        out_pil = Image.fromarray(np.asarray(out_pil)).convert("RGB")
    import numpy as np
    arr = np.asarray(out_pil)
    if arr.size and (arr.max() == arr.min() and arr.max() in (0, 255)):
        logger.warning("OmniGen output appears blank (min=%s max=%s). Try different seed or prompt.", arr.min(), arr.max())
    buf = io.BytesIO()
    out_pil.save(buf, format="PNG")
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

    # OmniGen(Omni) 사용 시: placeholder + input_images 로 이미지 편집 (LoRA 미지원)
    if _use_omnigen:
        compiled = ImagePromptExpert.compile(style_key, custom_prompt or "", aspect_ratio="1:1")
        prompt = compiled["final_prompt"] + (
            ", high detail, sharp focus, preserve original composition and subject."
        )
        num_steps_omni = max(1, min(50, num_steps or 50))
        loop = asyncio.get_event_loop()
        start = time.perf_counter()
        result = await loop.run_in_executor(
            None,
            lambda: _run_inference_omnigen_sync(
                image_bytes,
                prompt,
                seed,
                num_steps=num_steps_omni,
                guidance_scale=2.0,
                img_guidance_scale=1.6,
            ),
        )
        elapsed = time.perf_counter() - start
        return result, elapsed

    # Z-Image-Turbo: LoRA + strength/guidance/steps
    lora_filename = STYLE_TO_LORA_FILENAME.get(style_lower)
    lora_path = settings.lora_output_dir / lora_filename if lora_filename else None
    if lora_path and lora_path.exists():
        adapter_name = "lora_" + style_lower.replace(" ", "_")
        try:
            load_lora(pipe, lora_path, scale=LORA_SCALE, adapter_name=adapter_name)
            if hasattr(pipe, "set_adapters"):
                pipe.set_adapters([adapter_name])
            logger.info("LoRA loaded for style %s: %s", style_lower, lora_path)
        except Exception as e:
            logger.warning("LoRA load failed for %s (%s), using base model: %s", style_lower, lora_path, e)
            if hasattr(pipe, "set_adapters"):
                try:
                    pipe.set_adapters([])
                except Exception:
                    pass
    else:
        # 해당 스타일에 LoRA 없음 또는 파일 없음 → base 모델만 사용
        if hasattr(pipe, "set_adapters"):
            try:
                pipe.set_adapters([])
            except Exception:
                pass

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
    # style_presets 보강: 스타일별 네거티브 추가 (ImagePromptExpert + style_presets 동시 사용)
    try:
        style_presets_neg = get_style_negative_prompt(style_lower)
        if style_presets_neg:
            negative_prompt = f"{negative_prompt}, {style_presets_neg}"
    except Exception:
        pass
    if "pixel" in style_lower:
        negative_prompt = negative_prompt + PIXEL_ART_NEGATIVE_SUFFIX

    # 스타일별 strength 상한·기본값 (omni는 0.8까지 허용)
    default_st, max_st = STRENGTH_BY_STYLE.get(
        style_lower, (DEFAULT_STRENGTH_FALLBACK, STRENGTH_GLOBAL_MAX)
    )
    strength = strength if strength is not None else default_st
    strength_cap = OMNI_STRENGTH_MAX if style_lower == "omni" else STRENGTH_GLOBAL_MAX
    strength = max(0.0, min(strength_cap, min(1.0, strength), max_st))

    # Omni-Image-Editor: 50~70 steps, guidance 7~8 / 그 외: 기본값 또는 픽셀아트
    if style_lower == "omni":
        num_steps = max(1, min(OMNI_STEPS_MAX, num_steps or OMNI_NUM_STEPS))
        guidance_scale = OMNI_GUIDANCE_SCALE
    else:
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
