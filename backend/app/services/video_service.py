"""
LTX-2 Image-to-Video (Lightricks LTX-2).
사진 + 프롬프트 → 동영상 생성. Hugging Face diffusers LTX2ImageToVideoPipeline 사용.
기본: 768×512, 121 frames(약 5초), 25 steps, guidance 4.0 (ltx-2-TURBO 품질).
품질 모드: 241 frames(약 10초). Ref: https://huggingface.co/docs/diffusers/main/en/api/pipelines/ltx2
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

# ---------- ltx-2-TURBO 기준 (packages/ltx-pipelines/utils/constants.py) ----------
# 품질 모드: 768×512, 10초(241 frames), 25 steps, guidance 4.0
QUALITY_WIDTH = 768
QUALITY_HEIGHT = 512
QUALITY_NUM_FRAMES = 241  # 8n+1. 10초 @ 24fps = 241 (TURBO는 121=5초)
QUALITY_NUM_STEPS = 25
QUALITY_GUIDANCE_SCALE = 4.0
# TURBO DEFAULT_NEGATIVE_PROMPT (시네마틱 품질용) + 구도 고정 유도
NEGATIVE_PROMPT_TURBO = (
    "blurry, out of focus, overexposed, underexposed, low contrast, washed out colors, excessive noise, "
    "grainy texture, poor lighting, flickering, motion blur, distorted proportions, unnatural skin tones, "
    "deformed facial features, asymmetrical face, missing facial features, extra limbs, disfigured hands, "
    "wrong hand count, artifacts around text, inconsistent perspective, camera shake, incorrect depth of "
    "field, background too sharp, background clutter, distracting reflections, harsh shadows, inconsistent "
    "lighting direction, color banding, cartoonish rendering, 3D CGI look, unrealistic materials, uncanny "
    "valley effect, incorrect ethnicity, wrong gender, exaggerated expressions, wrong gaze direction, "
    "mismatched lip sync, silent or muted audio, distorted voice, robotic voice, echo, background noise, "
    "off-sync audio, incorrect dialogue, added dialogue, repetitive speech, jittery movement, awkward "
    "pauses, incorrect timing, unnatural transitions, inconsistent framing, tilted camera, flat lighting, "
    "inconsistent tone, cinematic oversaturation, stylized filters, or AI artifacts. "
    "camera movement, camera pan, panning shot, camera tilt, zoom in, zoom out, dolly, tracking shot, moving camera, crane shot."
)

# 기본: 최소 5초 영상 + TURBO 수준 품질 (768×512, 25 steps, guidance 4.0)
DEFAULT_WIDTH = 768
DEFAULT_HEIGHT = 512
DEFAULT_NUM_FRAMES = 121  # 8n+1. 5초 @ 24fps (33이면 1초대라 너무 짧음)
DEFAULT_FRAME_RATE = 24.0
DEFAULT_NUM_STEPS = 25
DEFAULT_GUIDANCE_SCALE = 4.0
# 기본 negative: TURBO와 동일 문구로 품질 유지 (짧은 버전은 품질 모드에서만 확장 가능하도록 생략)
DEFAULT_NEGATIVE = NEGATIVE_PROMPT_TURBO

# ---------- 반려동물 짧은 춤 영상 (3~4초, 속도·자연스러움 밸런스) ----------
# resolution 640x384, frames 25~33, steps 8~10, guidance 3.5, fps 8~10
DANCE_SHORT_WIDTH = 640
DANCE_SHORT_HEIGHT = 384
DANCE_SHORT_NUM_FRAMES = 33  # 8n+1. 8fps 기준 약 4초
DANCE_SHORT_NUM_STEPS = 8
DANCE_SHORT_GUIDANCE_SCALE = 3.5
DANCE_SHORT_FRAME_RATE = 8.0
# 반려동물 관절에 맞지 않아 프레임 깨짐 유발 → 네거티브에 추가
NEGATIVE_PET_DANCE = (
    "breakdance, acrobat, spin fast, jump high, backflip, somersault, "
    "extreme motion, unnatural pose, human dance, standing on two legs like human."
)


def _clamp_num_frames_to_8n_plus_1(n: int) -> int:
    """LTX-2는 num_frames가 8n+1 형태여야 함. 가장 가까운 유효값으로 보정."""
    if n <= 0:
        return 33
    # 8n+1: 33, 41, 49, 57, 65, 73, 81, ..., 121, 241
    remainder = (n - 1) % 8
    if remainder == 0:
        return max(33, min(241, n))
    # n보다 작은 최대 8k+1 또는 n보다 큰 최소 8k+1 중 가까운 쪽
    low = ((n - 1) // 8) * 8 + 1
    high = low + 8
    low = max(33, low)
    high = min(241, high)
    return low if (n - low) <= (high - n) else high


def _get_image_to_video_pipeline_class() -> tuple:
    """LTX2ImageToVideoPipeline 또는 LTX2ConditionPipeline 클래스 정보 반환."""
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


def _apply_cuda_optimizations() -> None:
    """H100 등에서 TF32·cudnn·matmul 정밀도 설정 (필수)."""
    import torch
    if not torch.cuda.is_available():
        return
    if hasattr(torch.backends, "cuda") and hasattr(torch.backends.cuda, "matmul"):
        torch.backends.cuda.matmul.allow_tf32 = True
    if hasattr(torch.backends, "cudnn"):
        torch.backends.cudnn.allow_tf32 = True
        torch.backends.cudnn.benchmark = True
    if hasattr(torch, "set_float32_matmul_precision"):
        torch.set_float32_matmul_precision("high")
    logger.info("LTX-2: CUDA tf32 / cudnn.benchmark / matmul high precision enabled")


def _load_pipeline_sync() -> Any:
    import os
    import torch

    pipeline_type = _get_image_to_video_pipeline_class()
    global _pipeline
    settings = get_settings()
    model_id = getattr(settings, "ltx2_model_id", "Lightricks/LTX-2")
    dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
    device = "cuda" if torch.cuda.is_available() else "cpu"
    # H100 등 VRAM 충분 시 CPU offload 제거로 대폭 속도 향상
    use_full_cuda = getattr(settings, "ltx2_use_full_cuda", True) and device == "cuda"

    _apply_cuda_optimizations()

    # Flash Attention 2 (로드 시 적용). pip install flash-attn
    load_kw: dict = {"torch_dtype": dtype}
    if device == "cuda" and getattr(settings, "enable_flash_attention_2", True):
        try:
            import importlib
            importlib.import_module("flash_attn")
            load_kw["attn_implementation"] = "flash_attention_2"
            logger.info("LTX-2: loading with attn_implementation=flash_attention_2")
        except Exception as e:
            logger.debug("LTX-2: Flash Attention 2 not available (%s)", e)

    if pipeline_type[0] == "image2video":
        PipeClass = pipeline_type[1]
        _pipeline = PipeClass.from_pretrained(model_id, **load_kw)
        if device == "cuda":
            if use_full_cuda:
                _pipeline = _pipeline.to(device)
                logger.info("LTX-2: full GPU (no CPU offload) for H100")
            else:
                _pipeline.enable_sequential_cpu_offload(device=device)
        else:
            _pipeline = _pipeline.to(device)
        logger.info("LTX-2 Image-to-Video pipeline loaded (model=%s)", model_id)
    else:
        PipeClass = pipeline_type[1]
        LTX2VideoCondition = pipeline_type[2]
        _pipeline = PipeClass.from_pretrained(model_id, **load_kw)
        if device == "cuda":
            if use_full_cuda:
                _pipeline = _pipeline.to(device)
                logger.info("LTX-2: full GPU (no CPU offload) for H100")
            else:
                _pipeline.enable_sequential_cpu_offload(device=device)
        else:
            _pipeline = _pipeline.to(device)
        if hasattr(_pipeline, "vae") and _pipeline.vae is not None:
            _pipeline.vae.enable_tiling()
        _pipeline._ltx_video_condition_class = LTX2VideoCondition
        logger.info("LTX-2 Condition pipeline loaded for image-to-video (model=%s)", model_id)

    # VAE decode 최적화: slicing + tiling (GPU 메모리·지연 감소, ~10–15% 개선)
    if hasattr(_pipeline, "enable_vae_slicing"):
        _pipeline.enable_vae_slicing()
        logger.info("LTX-2: VAE slicing enabled")
    if hasattr(_pipeline, "enable_vae_tiling"):
        _pipeline.enable_vae_tiling()
        logger.info("LTX-2: VAE tiling enabled")
    elif hasattr(_pipeline, "vae") and _pipeline.vae is not None and hasattr(_pipeline.vae, "enable_tiling"):
        _pipeline.vae.enable_tiling()
        logger.info("LTX-2: VAE tiling (vae.enable_tiling) enabled")

    # Attention: FA2 미사용 시 memory efficient attention
    if load_kw.get("attn_implementation") != "flash_attention_2":
        if hasattr(_pipeline, "enable_xformers_memory_efficient_attention"):
            try:
                _pipeline.enable_xformers_memory_efficient_attention()
                logger.info("LTX-2: xformers memory-efficient attention enabled")
            except Exception as e:
                logger.debug("LTX-2: xformers not available (%s)", e)
    # Flash Attention processor (일부 파이프라인은 로드 후에도 set 가능)
    if device == "cuda":
        for comp_name in ("transformer", "unet"):
            comp = getattr(_pipeline, comp_name, None)
            if comp is None or not hasattr(comp, "set_attn_processor"):
                continue
            try:
                from diffusers.models.attention_processor import FlashAttention2Processor
                comp.set_attn_processor(FlashAttention2Processor())
                logger.info("LTX-2: %s set_attn_processor(FlashAttention2Processor)", comp_name)
                break
            except Exception:
                pass

    # Scheduler: LTX-2는 Flow Matching 기반. DPM은 호환 안 될 수 있음 — 설정으로만 시도
    if getattr(settings, "ltx2_use_dpm_scheduler", False) and device == "cuda":
        try:
            from diffusers import DPMSolverMultistepScheduler
            _pipeline.scheduler = DPMSolverMultistepScheduler.from_config(_pipeline.scheduler.config)
            logger.info("LTX-2: DPMSolverMultistepScheduler applied (experimental)")
        except Exception as e:
            logger.warning("LTX-2: DPM scheduler skip, keeping default (%s)", e)

    # torch.compile: transformer 전체 (및 unet 있으면) max-autotune, fullgraph (목표 20–30% 개선)
    if device == "cuda":
        prev_disable = os.environ.get("TORCH_COMPILE_DISABLE")
        try:
            os.environ["TORCH_COMPILE_DISABLE"] = "0"
            for comp_name in ("transformer", "unet"):
                target = getattr(_pipeline, comp_name, None)
                if target is None:
                    continue
                try:
                    compiled = torch.compile(target, mode="max-autotune", fullgraph=True)
                    setattr(_pipeline, comp_name, compiled)
                    logger.info("LTX-2: torch.compile(%s, max-autotune, fullgraph=True)", comp_name)
                    break
                except Exception:
                    try:
                        compiled = torch.compile(target, mode="max-autotune", fullgraph=False)
                        setattr(_pipeline, comp_name, compiled)
                        logger.info("LTX-2: torch.compile(%s, max-autotune, fullgraph=False)", comp_name)
                        break
                    except Exception as e2:
                        try:
                            compiled = torch.compile(target, mode="reduce-overhead", fullgraph=False)
                            setattr(_pipeline, comp_name, compiled)
                            logger.info("LTX-2: torch.compile(%s, reduce-overhead)", comp_name)
                            break
                        except Exception as e3:
                            logger.debug("LTX-2: torch.compile(%s) skipped: %s", comp_name, e3)
        except Exception as e:
            logger.warning("LTX-2: torch.compile skipped (%s)", e)
        finally:
            if prev_disable is not None:
                os.environ["TORCH_COMPILE_DISABLE"] = prev_disable
            elif "TORCH_COMPILE_DISABLE" in os.environ:
                del os.environ["TORCH_COMPILE_DISABLE"]

    # Warmup: compile cache 생성 (num_frames=8, steps=4)
    if device == "cuda" and getattr(settings, "ltx2_warmup", True):
        try:
            from PIL import Image as PILImage
            warmup_img = PILImage.new("RGB", (320, 192), (128, 128, 128))
            with torch.inference_mode():
                if getattr(_pipeline, "_ltx_video_condition_class", None) is not None:
                    _cond = _pipeline._ltx_video_condition_class(frames=warmup_img, index=0, strength=1.0)
                    _pipeline(
                        conditions=[_cond],
                        prompt="warmup",
                        negative_prompt="",
                        width=320,
                        height=192,
                        num_frames=8,
                        frame_rate=24.0,
                        num_inference_steps=4,
                        guidance_scale=3.5,
                        output_type="latent",
                        return_dict=True,
                    )
                else:
                    _pipeline(
                        image=warmup_img,
                        prompt="warmup",
                        negative_prompt="",
                        width=320,
                        height=192,
                        num_frames=8,
                        frame_rate=24.0,
                        num_inference_steps=4,
                        guidance_scale=3.5,
                        output_type="latent",
                        return_dict=True,
                    )
            logger.info("LTX-2: warmup done (num_frames=8, steps=4, compile cache ready)")
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
    import numpy as np
    from PIL import Image, ImageOps

    global _pipeline
    if _pipeline is None:
        raise RuntimeError("Video pipeline not loaded")

    # 추론 파라미터 보정: num_frames는 8n+1 필수
    num_frames = _clamp_num_frames_to_8n_plus_1(num_frames)
    num_inference_steps = max(4, min(50, num_inference_steps))
    guidance_scale = max(1.0, min(10.0, guidance_scale))
    if not (prompt or "").strip():
        raise ValueError("prompt is required for image-to-video")

    img = Image.open(io.BytesIO(image_bytes))
    img = ImageOps.exif_transpose(img)
    img = img.convert("RGB")
    w, h = img.size
    scale_w = width / w
    scale_h = height / h
    scale = max(scale_w, scale_h)
    new_w = max(width, int(w * scale))
    new_h = max(height, int(h * scale))
    img = img.resize((new_w, new_h), Image.Resampling.LANCZOS)
    left = (new_w - width) // 2
    top = (new_h - height) // 2
    img = img.crop((left, top, left + width, top + height))

    # generator: 파이프라인 device와 맞추면 재현성·안정성에 유리
    device = "cuda" if torch.cuda.is_available() else "cpu"
    generator = None
    if seed is not None:
        try:
            generator = torch.Generator(device=device).manual_seed(seed)
        except Exception:
            generator = torch.Generator(device="cpu").manual_seed(seed)

    neg = (negative_prompt or "").strip() or DEFAULT_NEGATIVE
    logger.info(
        "LTX-2 image2video | %dx%d | frames=%d (8n+1) | steps=%d | cfg=%.1f",
        width, height, num_frames, num_inference_steps, guidance_scale,
    )

    use_condition = getattr(_pipeline, "_ltx_video_condition_class", None) is not None
    with torch.inference_mode():
        if use_condition:
            LTX2VideoCondition = _pipeline._ltx_video_condition_class
            condition = LTX2VideoCondition(frames=img, index=0, strength=1.0)
            out = _pipeline(
                conditions=[condition],
                prompt=prompt.strip(),
                negative_prompt=neg,
                width=width,
                height=height,
                num_frames=num_frames,
                frame_rate=frame_rate,
                num_inference_steps=num_inference_steps,
                guidance_scale=guidance_scale,
                output_type="np",
                return_dict=True,
                generator=generator,
            )
        else:
            out = _pipeline(
                image=img,
                prompt=prompt.strip(),
                negative_prompt=neg,
                width=width,
                height=height,
                num_frames=num_frames,
                frame_rate=frame_rate,
                num_inference_steps=num_inference_steps,
                guidance_scale=guidance_scale,
                output_type="np",
                return_dict=True,
                generator=generator,
            )

    # 반환: return_dict=True면 객체 속성, False면 (videos, audios) 튜플. 둘 다 안전 처리.
    if hasattr(out, "frames"):
        video_out = out.frames
    elif hasattr(out, "videos"):
        video_out = out.videos
    elif isinstance(out, (list, tuple)) and len(out) >= 1:
        video_out = out[0]
    else:
        video_out = out
    video_frames = video_out[0] if isinstance(video_out, (list, tuple)) else video_out
    # (batch, T, H, W, C) → (T, H, W, C); encode_video는 (T, H, W, C) 기대
    if hasattr(video_frames, "shape") and len(getattr(video_frames, "shape", ())) == 5:
        video_frames = video_frames[0]
    # torch.Tensor → numpy; 값 범위 [0,1] float → [0,255] uint8
    if hasattr(video_frames, "cpu"):
        video_frames = video_frames.float().cpu().numpy()
    if not isinstance(video_frames, np.ndarray):
        video_frames = np.asarray(video_frames)
    if video_frames.dtype != np.uint8:
        if video_frames.size > 0 and float(video_frames.max()) <= 1.0 and float(video_frames.min()) >= 0:
            video_frames = (np.clip(video_frames, 0.0, 1.0) * 255.0).astype(np.uint8)
        elif video_frames.dtype in (np.float32, np.float64):
            video_frames = np.clip(video_frames, 0, 255).astype(np.uint8)
    # (T, H, W, C) 형태 검증
    if len(video_frames.shape) != 4 or video_frames.shape[3] not in (3, 4):
        raise RuntimeError(
            f"LTX-2 pipeline returned invalid video shape: {getattr(video_frames, 'shape', 'unknown')}. "
            "Expected (num_frames, height, width, 3)."
        )
    if hasattr(out, "audios"):
        audio = out.audios
    elif isinstance(out, (list, tuple)) and len(out) >= 2:
        audio = out[1]
    else:
        audio = None

    from diffusers.pipelines.ltx2.export_utils import encode_video
    import tempfile
    import os

    with tempfile.NamedTemporaryFile(suffix=".mp4", delete=False) as f:
        tmp_path = f.name
    try:
        # encode_video(video, fps=..., output_path=..., audio=..., audio_sample_rate=...)
        audio_arg = None
        audio_sr = 24000
        if audio is not None and hasattr(_pipeline, "vocoder") and _pipeline.vocoder is not None:
            a_list = audio if isinstance(audio, (list, tuple)) else [audio]
            if len(a_list) > 0:
                audio_arg = a_list[0]
                if hasattr(audio_arg, "float"):
                    audio_arg = audio_arg.float().cpu()
                audio_sr = getattr(_pipeline.vocoder.config, "output_sampling_rate", 24000)
        encode_video(
            video_frames,
            fps=frame_rate,
            output_path=tmp_path,
            audio=audio_arg,
            audio_sample_rate=audio_sr,
        )
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
