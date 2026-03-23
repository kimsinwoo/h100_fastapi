"""
LTX-2 / LTX-2.3 Image-to-Video (Lightricks).
사진 + 프롬프트 → 동영상 생성. 기본은 LTX-2 + diffusers; LTX-2.3은 ComfyUI 사용.
- 해상도: 32 배수 필수. 프레임: 8n+1 (HF 카드).
- LTX-2: 2.3 스타일 품질 파이프라인 적용(품질 프리픽스, 텍스트 네거티브, ltx2_quality_mode 기본 True).
- LTX-2.3 distilled: 8 steps, CFG=1.0 권장. ComfyUI 사용 시 모델·파이프라인은 ComfyUI/models 및 pipelines 참조.
"""

from __future__ import annotations

import asyncio
import io
import logging
import time
from pathlib import Path
from typing import Any

from app.core.config import get_settings

logger = logging.getLogger(__name__)

_pipeline: Any = None
_pipeline_lock = asyncio.Lock()

# ---------- LTX-2.3-22b 기준 (HF 카드: 해상도 32배수, 프레임 8n+1, distilled 8 steps CFG=1) ----------
# 품질 모드: 768×432(32배수), 10초(241 frames), 25 steps (full) / 8 steps (distilled)
QUALITY_WIDTH = 768
QUALITY_HEIGHT = 512   # 32*16
QUALITY_NUM_FRAMES = 241  # 8n+1. 10초 @ 24fps
QUALITY_NUM_STEPS = 25   # full; distilled는 8
QUALITY_GUIDANCE_SCALE = 4.0  # full; distilled는 1.0
# LTX-2.3 distilled 기본값 (성능·품질 균형)
LTX23_DEFAULT_STEPS = 8
LTX23_DEFAULT_GUIDANCE = 1.0
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

# 기본: 32배수 해상도, 8n+1 프레임. LTX-2.3일 때 steps/guidance는 config ltx23_* 사용
DEFAULT_WIDTH = 768   # 32*24
DEFAULT_HEIGHT = 512  # 32*16
DEFAULT_NUM_FRAMES = 121  # 8n+1. 5초 @ 24fps
DEFAULT_FRAME_RATE = 24.0
DEFAULT_NUM_STEPS = 25
DEFAULT_GUIDANCE_SCALE = 4.0
# 기본 negative: TURBO와 동일 문구로 품질 유지 (짧은 버전은 품질 모드에서만 확장 가능하도록 생략)
DEFAULT_NEGATIVE = NEGATIVE_PROMPT_TURBO

# ---------- LTX-2 품질 파이프라인 (2.3 스타일 텍스트·시네마틱 품질) ----------
# 프롬프트 앞에 붙여 시네마틱/디테일 유도 (LTX-2 diffusers 경로)
VIDEO_PROMPT_QUALITY_PREFIX = (
    "High quality, cinematic, detailed motion, natural lighting, sharp focus. "
)
# 텍스트/자막 관련 네거티브 확장 (품질 파이프라인에서 기본 네거티브에 추가)
NEGATIVE_VIDEO_TEXT = (
    "floating text, wrong text, distorted letters, watermark, subtitle artifacts, "
    "burned-in text, logo, caption errors, blurry text, text overlay, floating captions."
)

# ---------- 반려동물 영상 (모션 안정성: 스피드맨/일그러짐 방지) ----------
# motion strength 0.35~0.50: 0.60+면 움직임 과장·관절 폭발·스피드맨. guidance 2.0~2.8: 4+면 과장 동작.
# frames 25~41 권장, 60+면 구조 drift. resolution 640, fps 12 권장.
DANCE_SHORT_WIDTH = 640
DANCE_SHORT_HEIGHT = 384
DANCE_SHORT_NUM_FRAMES = 33  # 8n+1. 25~41 구간에서 가장 안정
DANCE_SHORT_NUM_STEPS = 8
DANCE_SHORT_GUIDANCE_SCALE = 2.3  # 2.0~2.8. 높으면 sway/steps/dance 과장
DANCE_SHORT_FRAME_RATE = 12.0
# 동물 모델에서 0.60+는 거의 깨짐. 0.35~0.50 권장
DANCE_CONDITION_STRENGTH = 0.42
# 반드시 넣어야 하는 negative motion (없으면 과한 앞발·흔들림)
NEGATIVE_PET_DANCE = (
    "no aggressive paw movement, no fast shaking, no chaotic movement, no exaggerated motion, "
    "no wild paw waving, no wild movement, no chaotic paw waving, no uncontrolled motion, "
    "breakdance, acrobat, spin fast, jump high, backflip, somersault, "
    "extreme motion, unnatural pose, human dance, standing on two legs like human."
)


def _clamp_resolution_to_32(value: int, min_val: int = 32, max_val: int = 1280) -> int:
    """LTX-2.3: width/height는 32 배수여야 함 (HF 카드)."""
    if value <= 0:
        return min_val
    base = max(min_val, min(max_val, (value // 32) * 32))
    return base if base >= 32 else 32


def _is_ltx23() -> bool:
    """설정의 모델 ID가 LTX-2.3 계열인지."""
    model_id = getattr(get_settings(), "ltx2_model_id", "") or ""
    return "2.3" in model_id or "ltx-2.3" in model_id.lower()


def _clamp_num_frames_to_8n_plus_1(n: int, min_frames: int = 25) -> int:
    """LTX-2/2.3: num_frames는 8n+1 형태 필수. 가장 가까운 유효값으로 보정.
    min_frames: 춤 영상은 25(24~28), 일반 영상은 33 이상 권장."""
    if n <= 0:
        return max(25, min_frames) if min_frames <= 25 else 33
    # 8n+1: 25, 33, 41, 49, ... (25=8*3+1 유효)
    remainder = (n - 1) % 8
    if remainder == 0:
        return max(min_frames, min(241, n))
    low = ((n - 1) // 8) * 8 + 1
    high = low + 8
    low = max(min_frames, low)
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
        try:
            _pipeline = PipeClass.from_pretrained(model_id, **load_kw)
        except Exception as e:
            if "2.3" in model_id or "ltx-2.3" in model_id.lower():
                raise RuntimeError(
                    f"LTX-2.3 ({model_id}) is not yet available in this diffusers version. "
                    "Use ComfyUI: set LTX2_USE_COMFYUI=true and add pipelines/ltx23_i2v.json (export from ComfyUI LTXVideo nodes)."
                ) from e
            raise
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
        try:
            _pipeline = PipeClass.from_pretrained(model_id, **load_kw)
        except Exception as e:
            if "2.3" in model_id or "ltx-2.3" in model_id.lower():
                raise RuntimeError(
                    f"LTX-2.3 ({model_id}) is not yet available in this diffusers version. "
                    "Use ComfyUI: set LTX2_USE_COMFYUI=true and add pipelines/ltx23_i2v.json (export from ComfyUI LTXVideo nodes)."
                ) from e
            raise
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
    condition_strength: float = 1.0,
) -> bytes:
    import torch
    import numpy as np
    from PIL import Image, ImageOps

    global _pipeline
    if _pipeline is None:
        raise RuntimeError("Video pipeline not loaded")

    # LTX-2.3: 해상도 32배수, 프레임 8n+1 필수
    width = _clamp_resolution_to_32(width)
    height = _clamp_resolution_to_32(height)
    num_frames = _clamp_num_frames_to_8n_plus_1(num_frames)
    num_inference_steps = max(4, min(50, num_inference_steps))
    guidance_scale = max(0.0, min(10.0, guidance_scale))
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
    strength = max(0.0, min(1.0, condition_strength))
    with torch.inference_mode():
        if use_condition:
            LTX2VideoCondition = _pipeline._ltx_video_condition_class
            condition = LTX2VideoCondition(frames=img, index=0, strength=strength)
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
    condition_strength: float | None = None,
    reference_video_path: Path | None = None,
) -> tuple[bytes, float]:
    settings = get_settings()
    # LTX-2.3-22b distilled: config 기본값 사용 (8 steps, CFG=1)
    if _is_ltx23():
        if num_inference_steps == DEFAULT_NUM_STEPS:
            num_inference_steps = getattr(settings, "ltx23_num_steps", LTX23_DEFAULT_STEPS)
        if guidance_scale == DEFAULT_GUIDANCE_SCALE:
            guidance_scale = getattr(settings, "ltx23_guidance_scale", LTX23_DEFAULT_GUIDANCE)

    # ComfyUI 사용 조건: LTX2_USE_COMFYUI=true 이거나, ComfyUI 활성 + 워크플로 파일 존재 시 (품질용 ComfyUI JSON 있으면 우선 사용)
    use_comfyui = getattr(settings, "ltx2_use_comfyui", False)
    if not use_comfyui and getattr(settings, "comfyui_enabled", False):
        wf_name = getattr(settings, "comfyui_ltx23_workflow", "ltx23_i2v") or "ltx23_i2v"
        wf_path = settings.pipelines_dir / f"{wf_name}.json"
        if not wf_path.exists():
            wf_path = settings.pipelines_dir / "comfyui_ltx23_workflow.json"
        if wf_path.exists():
            use_comfyui = True
            logger.info("Using ComfyUI for video (workflow found: %s)", wf_path.name)
    if _is_ltx23() and not use_comfyui:
        raise RuntimeError(
            "LTX-2.3 is not supported by this diffusers version. "
            "Use ComfyUI: set LTX2_USE_COMFYUI=true or COMFYUI_ENABLED=true and add pipelines/ltx23_i2v.json or comfyui_ltx23_workflow.json. "
            "See backend/pipelines/README_LTX23.md"
        )

    # Wan VACE v2v: 이미지 + 레퍼런스 영상 → 영상 (pipelines/video_wan_vace_14B_v2v.json 등)
    wan_enabled = getattr(settings, "wan_vace_v2v_enabled", False)
    wan_name = getattr(settings, "comfyui_wan_vace_v2v_workflow", "video_wan_vace_14B_v2v") or "video_wan_vace_14B_v2v"
    wan_path = settings.pipelines_dir / f"{wan_name}.json"
    if (
        wan_enabled
        and getattr(settings, "comfyui_enabled", False)
        and reference_video_path
        and reference_video_path.exists()
        and wan_path.is_file()
    ):
        from app.services.comfyui_service import run_wan_vace_v2v_image_to_video as _run_wan_vace

        start = time.perf_counter()
        out_bytes = await _run_wan_vace(
            image_bytes=image_bytes,
            prompt=prompt,
            negative_prompt=negative_prompt or DEFAULT_NEGATIVE,
            reference_video_path=reference_video_path,
            seed=seed,
            width=width,
            height=height,
            num_frames=num_frames,
        )
        return out_bytes, time.perf_counter() - start

    if use_comfyui:
        from app.services.comfyui_service import run_ltx23_image_to_video as _run_comfyui
        start = time.perf_counter()
        out_bytes = await _run_comfyui(
            image_bytes=image_bytes,
            prompt=prompt,
            negative_prompt=negative_prompt or DEFAULT_NEGATIVE,
            width=width,
            height=height,
            num_frames=num_frames,
            frame_rate=frame_rate,
            num_inference_steps=num_inference_steps,
            guidance_scale=guidance_scale,
            seed=seed,
            reference_video_path=reference_video_path,
        )
        return out_bytes, time.perf_counter() - start

    # LTX-2 diffusers: 2.3 스타일 품질 파이프라인 적용 (텍스트·시네마틱)
    prompt_quality = (VIDEO_PROMPT_QUALITY_PREFIX + prompt.strip()) if (prompt or "").strip() else prompt
    neg = (negative_prompt or "").strip() or DEFAULT_NEGATIVE
    if neg == DEFAULT_NEGATIVE or not neg:
        neg = f"{DEFAULT_NEGATIVE} {NEGATIVE_VIDEO_TEXT}".strip()

    strength = condition_strength if condition_strength is not None else 1.0
    await get_video_pipeline()
    loop = asyncio.get_event_loop()
    start = time.perf_counter()
    result = await loop.run_in_executor(
        None,
        lambda: _run_image_to_video_sync(
            image_bytes,
            prompt_quality,
            neg,
            width,
            height,
            num_frames,
            frame_rate,
            num_inference_steps,
            guidance_scale,
            seed,
            condition_strength=strength,
        ),
    )
    elapsed = time.perf_counter() - start
    return result, elapsed


def is_video_pipeline_loaded() -> bool:
    return _pipeline is not None
