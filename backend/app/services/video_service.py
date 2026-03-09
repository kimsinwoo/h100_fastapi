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

# 기본 해상도/프레임 (빠른 생성: 640×384, 33프레임, 12스텝, guidance 3.5)
DEFAULT_WIDTH = 640
DEFAULT_HEIGHT = 384
DEFAULT_NUM_FRAMES = 33  # 8n+1
DEFAULT_FRAME_RATE = 24.0
DEFAULT_NUM_STEPS = 12
DEFAULT_GUIDANCE_SCALE = 3.5
DEFAULT_NEGATIVE = (
    "worst quality, inconsistent motion, blurry, jittery, distorted, "
    "shaky, glitchy, deformed, motion smear, motion artifacts, bad anatomy, static."
)


def _get_image_to_video_pipeline_class() -> tuple:
    """LTX2ImageToVideoPipeline 또는 LTX2ConditionPipeline 클래스 정보 반환. 없으면 ImportError."""
    try:
        from diffusers import LTX2ImageToVideoPipeline
        return ("image2video", LTX2ImageToVideoPipeline)
    except ImportError:
        pass
    try:
        from diffusers.pipelines.ltx2 import LTX2ImageToVideoPipeline
        return ("image2video", LTX2ImageToVideoPipeline)
    except ImportError:
        pass
    try:
        from diffusers.pipelines.ltx2.pipeline_ltx2_image2video import LTX2ImageToVideoPipeline
        return ("image2video", LTX2ImageToVideoPipeline)
    except ImportError:
        pass
    try:
        from diffusers import LTX2ConditionPipeline
        from diffusers.pipelines.ltx2.pipeline_ltx2_condition import LTX2VideoCondition
        return ("condition", LTX2ConditionPipeline, LTX2VideoCondition)
    except ImportError:
        try:
            from diffusers.pipelines.ltx2 import LTX2ConditionPipeline
            from diffusers.pipelines.ltx2.pipeline_ltx2_condition import LTX2VideoCondition
            return ("condition", LTX2ConditionPipeline, LTX2VideoCondition)
        except ImportError as e:
            raise ImportError(
                "LTX-2 비디오 파이프라인을 불러올 수 없습니다. diffusers main 브랜치 필요. "
                "설치: pip install git+https://github.com/huggingface/diffusers.git"
            ) from e


def _load_pipeline_sync() -> Any:
    import os
    import torch

    pipeline_type = _get_image_to_video_pipeline_class()
    global _pipeline
    model_id = getattr(get_settings(), "ltx2_model_id", "Lightricks/LTX-2")
    dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
    device = "cuda" if torch.cuda.is_available() else "cpu"

    # CUDA 속도 최적화 (1분 미만 목표)
    if device == "cuda":
        if hasattr(torch.backends, "cuda") and hasattr(torch.backends.cuda, "matmul"):
            torch.backends.cuda.matmul.allow_tf32 = True
        if hasattr(torch.backends, "cudnn"):
            torch.backends.cudnn.benchmark = True
        logger.info("LTX-2: CUDA tf32/cudnn.benchmark enabled")

    # Flash Attention 2 (로드 시 적용 시 속도·메모리 최적화). pip install flash-attn
    load_kw: dict = {"torch_dtype": dtype}
    if device == "cuda" and getattr(get_settings(), "enable_flash_attention_2", True):
        try:
            import importlib
            importlib.import_module("flash_attn")
            load_kw["attn_implementation"] = "flash_attention_2"
            logger.info("LTX-2: loading with attn_implementation=flash_attention_2")
        except Exception as e:
            logger.debug("LTX-2: Flash Attention 2 not available (%s), using default", e)

    if pipeline_type[0] == "image2video":
        PipeClass = pipeline_type[1]
        _pipeline = PipeClass.from_pretrained(model_id, **load_kw)
        if device == "cuda":
            _pipeline.enable_sequential_cpu_offload(device=device)
        else:
            _pipeline = _pipeline.to(device)
        logger.info("LTX-2 Image-to-Video pipeline loaded (model=%s)", model_id)
    else:
        PipeClass = pipeline_type[1]
        LTX2VideoCondition = pipeline_type[2]
        _pipeline = PipeClass.from_pretrained(model_id, **load_kw)
        if device == "cuda":
            _pipeline.enable_sequential_cpu_offload(device=device)
        else:
            _pipeline = _pipeline.to(device)
        if hasattr(_pipeline, "vae") and _pipeline.vae is not None:
            _pipeline.vae.enable_tiling()
        _pipeline._ltx_video_condition_class = LTX2VideoCondition
        logger.info("LTX-2 Condition pipeline loaded for image-to-video (model=%s)", model_id)

    # VAE slicing: 디코드 시 메모리·처리 경량화 (비디오 프레임 많아서 유리)
    if hasattr(_pipeline, "enable_vae_slicing"):
        _pipeline.enable_vae_slicing()
        logger.info("LTX-2: VAE slicing enabled")
    # Flash Attention 2 미사용 시에만 xformers 폴백 (FA2는 로드 시 이미 적용됨)
    if "attn_implementation" not in load_kw or load_kw.get("attn_implementation") != "flash_attention_2":
        if hasattr(_pipeline, "enable_xformers_memory_efficient_attention"):
            try:
                _pipeline.enable_xformers_memory_efficient_attention()
                logger.info("LTX-2: xformers memory-efficient attention enabled")
            except Exception as e:
                logger.debug("LTX-2: xformers not available (%s), using default SDPA", e)

    # torch.compile: diffusion step당 계산 가속 (목표 ~40% 단축)
    if device == "cuda":
        target = getattr(_pipeline, "transformer", None)
        if target is not None:
            prev_disable = os.environ.get("TORCH_COMPILE_DISABLE")
            try:
                os.environ["TORCH_COMPILE_DISABLE"] = "0"
                try:
                    compiled = torch.compile(target, mode="max-autotune", fullgraph=True)
                    _pipeline.transformer = compiled
                    logger.info("LTX-2: torch.compile(transformer, max-autotune, fullgraph=True) enabled")
                except Exception:
                    try:
                        compiled = torch.compile(target, mode="max-autotune", fullgraph=False)
                        _pipeline.transformer = compiled
                        logger.info("LTX-2: torch.compile(transformer, max-autotune, fullgraph=False) enabled")
                    except Exception as e2:
                        compiled = torch.compile(target, mode="reduce-overhead", fullgraph=False)
                        _pipeline.transformer = compiled
                        logger.info("LTX-2: torch.compile(transformer, reduce-overhead) enabled")
            except Exception as e:
                logger.warning("LTX-2: torch.compile skipped (%s)", e)
            finally:
                if prev_disable is not None:
                    os.environ["TORCH_COMPILE_DISABLE"] = prev_disable
                elif "TORCH_COMPILE_DISABLE" in os.environ:
                    del os.environ["TORCH_COMPILE_DISABLE"]

    # 1회 워밍업: torch.compile 첫 실행 시 컴파일되어, 이후 요청부터 가속 적용
    if device == "cuda" and getattr(get_settings(), "ltx2_warmup", True):
        try:
            from PIL import Image as PILImage
            warmup_img = PILImage.new("RGB", (320, 192), (128, 128, 128))
            with torch.inference_mode():
                if getattr(_pipeline, "_ltx_video_condition_class", None) is not None:
                    _cond = _pipeline._ltx_video_condition_class(frames=warmup_img, index=0, strength=1.0)
                    _pipeline(conditions=[_cond], prompt="warmup", negative_prompt="", width=320, height=192, num_frames=17, frame_rate=24.0, num_inference_steps=2, guidance_scale=3.5, output_type="latent", return_dict=False)
                else:
                    _pipeline(image=warmup_img, prompt="warmup", negative_prompt="", width=320, height=192, num_frames=17, frame_rate=24.0, num_inference_steps=2, guidance_scale=3.5, output_type="latent", return_dict=False)
            logger.info("LTX-2: warmup done (compile cache ready)")
        except Exception as w:
            logger.debug("LTX-2: warmup skipped (%s)", w)

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

    use_condition = getattr(_pipeline, "_ltx_video_condition_class", None) is not None
    with torch.inference_mode():
        if use_condition:
            LTX2VideoCondition = _pipeline._ltx_video_condition_class
            condition = LTX2VideoCondition(frames=img, index=0, strength=1.0)
            out = _pipeline(
                conditions=[condition],
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
        else:
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

    # out = (video, audio); video는 배치 리스트이므로 video[0]이 [T,H,W,C] numpy
    video_out = out[0]
    video_frames = video_out[0] if isinstance(video_out, (list, tuple)) else video_out
    audio = out[1] if len(out) > 1 else None

    from diffusers.pipelines.ltx2.export_utils import encode_video
    import tempfile
    import os

    with tempfile.NamedTemporaryFile(suffix=".mp4", delete=False) as f:
        tmp_path = f.name
    try:
        kwargs = {"output_path": tmp_path, "fps": frame_rate}
        if audio is not None and len(audio) > 0 and hasattr(_pipeline, "vocoder") and _pipeline.vocoder is not None:
            audio_tensor = audio[0] if isinstance(audio, (list, tuple)) else audio
            if hasattr(audio_tensor, "float"):
                kwargs["audio"] = audio_tensor.float().cpu()
            kwargs["audio_sample_rate"] = getattr(
                _pipeline.vocoder.config, "output_sampling_rate", 24000
            )
        encode_video(video_frames, **kwargs)
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
