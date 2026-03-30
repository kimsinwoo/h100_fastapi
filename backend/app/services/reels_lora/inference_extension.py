"""
Generation pipeline extension: dynamic LoRA injection for Reels dance generation.

Integrates with the existing video generation pipeline without modifying it.

Two modes:
  1. AnimateDiff mode  — uses trained AnimateDiff LoRA directly
  2. LTX-2 passthrough — falls back to existing run_image_to_video() with
                         an enhanced prompt derived from the selected category

The AnimateDiff pipeline is loaded lazily and shared across requests.
LoRA weights are hot-swapped per request via PEFT set_adapter / load_adapter.
"""

from __future__ import annotations

import asyncio
import io
import logging
import threading
import time
from pathlib import Path
from typing import Optional, Tuple

import torch

from .lora_registry import (
    LoRAEntry,
    LoRARegistry,
    get_registry,
    NEGATIVE_PROMPT,
)

logger = logging.getLogger(__name__)

# ──────────────────────────────────────────────────────────────────────────────
# Generation parameters
# ──────────────────────────────────────────────────────────────────────────────

REELS_WIDTH = 512
REELS_HEIGHT = 512
REELS_FRAMES = 16
REELS_FPS = 8
REELS_STEPS = 25
REELS_GUIDANCE = 7.5
REELS_LORA_SCALE = 0.8     # LoRA influence (0.0 = off, 1.0 = full)


# ──────────────────────────────────────────────────────────────────────────────
# AnimateDiff pipeline manager
# ──────────────────────────────────────────────────────────────────────────────

class _AnimateDiffPipelineManager:
    """
    Singleton that owns one AnimateDiff pipeline.
    LoRA weights are swapped in/out per generation request.
    Thread-safe via asyncio semaphore (one generation at a time).
    """

    def __init__(self):
        self._pipe = None
        self._current_lora: Optional[str] = None
        self._init_lock = threading.Lock()
        self._gpu_sem = asyncio.Semaphore(1)

    # ──────────────────────────────────────────────────────────────────────────
    def _load_pipeline(
        self,
        base_model: str = "runwayml/stable-diffusion-v1-5",
        motion_adapter: str = "guoyww/animatediff-motion-adapter-v1-5-2",
        device: str = "cuda",
        dtype: torch.dtype = torch.bfloat16,
    ):
        try:
            from diffusers import AnimateDiffPipeline, MotionAdapter, DDIMScheduler
        except ImportError:
            raise RuntimeError(
                "diffusers>=0.25.0 required. pip install diffusers>=0.25.0"
            )

        logger.info("[AnimateDiff] Loading motion adapter: %s", motion_adapter)
        adapter = MotionAdapter.from_pretrained(
            motion_adapter,
            torch_dtype=dtype,
        )

        logger.info("[AnimateDiff] Loading pipeline: %s", base_model)
        scheduler = DDIMScheduler.from_pretrained(
            base_model,
            subfolder="scheduler",
            clip_sample=False,
            timestep_spacing="linspace",
            beta_schedule="linear",
            steps_offset=1,
        )
        pipe = AnimateDiffPipeline.from_pretrained(
            base_model,
            motion_adapter=adapter,
            scheduler=scheduler,
            torch_dtype=dtype,
        )
        pipe = pipe.to(device)

        # Memory optimizations
        try:
            pipe.enable_xformers_memory_efficient_attention()
        except Exception:
            pass

        pipe.enable_vae_slicing()
        logger.info("[AnimateDiff] Pipeline ready on %s", device)
        return pipe

    # ──────────────────────────────────────────────────────────────────────────
    def get_pipeline(self, entry: LoRAEntry):
        """Lazy-init pipeline and return it. Loads LoRA if needed."""
        with self._init_lock:
            device = "cuda" if torch.cuda.is_available() else "cpu"
            dtype = torch.bfloat16 if device == "cuda" else torch.float32

            if self._pipe is None:
                self._pipe = self._load_pipeline(
                    base_model=entry.base_model,
                    motion_adapter=entry.motion_adapter,
                    device=device,
                    dtype=dtype,
                )

            # Hot-swap LoRA if different category requested
            if self._current_lora != entry.category:
                self._apply_lora(entry, device, dtype)
                self._current_lora = entry.category

        return self._pipe

    # ──────────────────────────────────────────────────────────────────────────
    def _apply_lora(
        self,
        entry: LoRAEntry,
        device: str,
        dtype: torch.dtype,
    ) -> None:
        """Load and apply LoRA weights into the UNet."""
        if not entry.weights_path.exists():
            logger.warning("[AnimateDiff] LoRA weights not found: %s", entry.weights_path)
            return

        try:
            self._pipe.load_lora_weights(
                str(entry.weights_path.parent),
                weight_name="lora_weights.safetensors",
                adapter_name=entry.category,
            )
            self._pipe.set_adapters([entry.category], adapter_weights=[REELS_LORA_SCALE])
            logger.info("[AnimateDiff] LoRA '%s' applied (scale=%.2f)", entry.category, REELS_LORA_SCALE)
        except Exception as e:
            logger.error("[AnimateDiff] LoRA apply failed for '%s': %s", entry.category, e)

    # ──────────────────────────────────────────────────────────────────────────
    async def generate(
        self,
        entry: LoRAEntry,
        prompt: str,
        negative_prompt: str,
        image_bytes: bytes,
        num_frames: int = REELS_FRAMES,
        num_steps: int = REELS_STEPS,
        guidance_scale: float = REELS_GUIDANCE,
        width: int = REELS_WIDTH,
        height: int = REELS_HEIGHT,
        seed: Optional[int] = None,
    ) -> Tuple[bytes, float]:
        """
        Generate a dance video with the specified LoRA.
        Returns (mp4_bytes, elapsed_seconds).
        """
        async with self._gpu_sem:
            loop = asyncio.get_event_loop()
            result = await loop.run_in_executor(
                None,
                self._generate_sync,
                entry, prompt, negative_prompt, image_bytes,
                num_frames, num_steps, guidance_scale,
                width, height, seed,
            )
        return result

    # ──────────────────────────────────────────────────────────────────────────
    def _generate_sync(
        self,
        entry: LoRAEntry,
        prompt: str,
        negative_prompt: str,
        image_bytes: bytes,
        num_frames: int,
        num_steps: int,
        guidance_scale: float,
        width: int,
        height: int,
        seed: Optional[int],
    ) -> Tuple[bytes, float]:
        from PIL import Image

        t0 = time.perf_counter()
        pipe = self.get_pipeline(entry)

        # Decode input image
        pil_image = Image.open(io.BytesIO(image_bytes)).convert("RGB")
        pil_image = pil_image.resize((width, height), Image.LANCZOS)

        # Generator
        generator = None
        if seed is not None:
            generator = torch.Generator(device=pipe.device.type).manual_seed(seed)

        # Inference
        output = pipe(
            prompt=prompt,
            negative_prompt=negative_prompt,
            num_frames=num_frames,
            num_inference_steps=num_steps,
            guidance_scale=guidance_scale,
            width=width,
            height=height,
            generator=generator,
        )

        frames = output.frames[0]  # list of PIL Images

        # Encode to mp4
        mp4_bytes = _encode_frames_to_mp4(frames, fps=REELS_FPS)
        elapsed = time.perf_counter() - t0

        logger.info(
            "[AnimateDiff] Generated %d frames in %.1fs (category=%s)",
            len(frames), elapsed, entry.category,
        )
        return mp4_bytes, elapsed


# ──────────────────────────────────────────────────────────────────────────────
# MP4 encoder
# ──────────────────────────────────────────────────────────────────────────────

def _encode_frames_to_mp4(frames, fps: int = 8) -> bytes:
    """Encode a list of PIL Images to mp4 bytes using av (pyav)."""
    import av
    import numpy as np

    buf = io.BytesIO()
    container = av.open(buf, mode="w", format="mp4")
    stream = container.add_stream("h264", rate=fps)
    stream.width = frames[0].width
    stream.height = frames[0].height
    stream.pix_fmt = "yuv420p"
    stream.options = {"crf": "23", "preset": "fast"}

    for pil_img in frames:
        arr = np.array(pil_img.convert("RGB"))
        frame = av.VideoFrame.from_ndarray(arr, format="rgb24")
        frame = frame.reformat(format=stream.pix_fmt)
        for pkt in stream.encode(frame):
            container.mux(pkt)

    for pkt in stream.encode():
        container.mux(pkt)

    container.close()
    return buf.getvalue()


# ──────────────────────────────────────────────────────────────────────────────
# Public interface
# ──────────────────────────────────────────────────────────────────────────────

_pipeline_manager: Optional[_AnimateDiffPipelineManager] = None
_pm_lock = threading.Lock()


def _get_pipeline_manager() -> _AnimateDiffPipelineManager:
    global _pipeline_manager
    with _pm_lock:
        if _pipeline_manager is None:
            _pipeline_manager = _AnimateDiffPipelineManager()
    return _pipeline_manager


class ReelsDanceGenerator:
    """
    High-level generator that selects the right LoRA and produces a dance video.

    Supports two modes:
      - "animatediff": Use trained AnimateDiff LoRA (requires trained weights)
      - "ltx2":        Use existing LTX-2 pipeline with enhanced prompt

    Usage:
        gen = ReelsDanceGenerator()
        video_bytes, elapsed = await gen.generate(
            image_bytes=...,
            category="tiktok_shuffle",
            character="dog",
            mode="animatediff",
        )
    """

    def __init__(self, registry: Optional[LoRARegistry] = None):
        self.registry = registry or get_registry()

    async def generate(
        self,
        image_bytes: bytes,
        category: str,
        character: str = "dog",
        mode: str = "ltx2",          # "animatediff" | "ltx2"
        seed: Optional[int] = None,
        num_frames: int = REELS_FRAMES,
        num_steps: Optional[int] = None,
        width: int = REELS_WIDTH,
        height: int = REELS_HEIGHT,
    ) -> Tuple[bytes, float]:
        """
        Generate a dance video for the given category.

        Args:
            image_bytes: Input character image (JPEG/PNG bytes)
            category:    Dance category (e.g. "tiktok_shuffle")
            character:   Character description (e.g. "dog", "cat")
            mode:        "animatediff" uses trained LoRA; "ltx2" uses existing pipeline
            seed:        Optional random seed
            num_frames:  Number of output frames
            num_steps:   Inference steps (defaults: AnimateDiff=25, LTX-2=8)
            width/height: Output resolution

        Returns:
            (mp4_bytes, elapsed_seconds)
        """
        prompt = self.registry.get_prompt(category, character)
        neg = self.registry.get_negative_prompt()

        if mode == "animatediff":
            return await self._generate_animatediff(
                image_bytes=image_bytes,
                category=category,
                prompt=prompt,
                negative_prompt=neg,
                seed=seed,
                num_frames=num_frames,
                num_steps=num_steps or REELS_STEPS,
                width=width,
                height=height,
            )
        else:
            return await self._generate_ltx2_passthrough(
                image_bytes=image_bytes,
                prompt=prompt,
                negative_prompt=neg,
                seed=seed,
                num_frames=num_frames,
                num_steps=num_steps,
                width=width,
                height=height,
            )

    # ──────────────────────────────────────────────────────────────────────────
    async def _generate_animatediff(
        self,
        image_bytes: bytes,
        category: str,
        prompt: str,
        negative_prompt: str,
        seed: Optional[int],
        num_frames: int,
        num_steps: int,
        width: int,
        height: int,
    ) -> Tuple[bytes, float]:
        entry = self.registry.get(category)
        if entry is None:
            logger.warning(
                "[Generator] LoRA '%s' not in registry — falling back to LTX-2",
                category,
            )
            return await self._generate_ltx2_passthrough(
                image_bytes=image_bytes,
                prompt=prompt,
                negative_prompt=negative_prompt,
                seed=seed,
                num_frames=num_frames,
                num_steps=num_steps,
                width=width,
                height=height,
            )

        if not entry.weights_path.exists():
            logger.warning(
                "[Generator] LoRA weights missing for '%s' — falling back to LTX-2",
                category,
            )
            return await self._generate_ltx2_passthrough(
                image_bytes=image_bytes,
                prompt=prompt,
                negative_prompt=negative_prompt,
                seed=seed,
                num_frames=num_frames,
                num_steps=num_steps,
                width=width,
                height=height,
            )

        manager = _get_pipeline_manager()
        return await manager.generate(
            entry=entry,
            prompt=prompt,
            negative_prompt=negative_prompt,
            image_bytes=image_bytes,
            num_frames=num_frames,
            num_steps=num_steps,
            guidance_scale=REELS_GUIDANCE,
            width=width,
            height=height,
            seed=seed,
        )

    # ──────────────────────────────────────────────────────────────────────────
    async def _generate_ltx2_passthrough(
        self,
        image_bytes: bytes,
        prompt: str,
        negative_prompt: str,
        seed: Optional[int],
        num_frames: Optional[int],
        num_steps: Optional[int],
        width: int,
        height: int,
    ) -> Tuple[bytes, float]:
        """
        Delegate to the existing LTX-2 video pipeline with reels-optimized prompt.
        Keeps all existing LTX-2 behaviour intact.
        """
        from app.services.video_service import (
            compute_i2v_output_dimensions_from_bytes,
            run_image_to_video,
            DANCE_SHORT_NUM_FRAMES,
            DANCE_SHORT_FRAME_RATE,
            DANCE_SHORT_NUM_STEPS,
            DANCE_SHORT_GUIDANCE_SCALE,
            DANCE_CONDITION_STRENGTH,
            NEGATIVE_PET_DANCE,
        )

        dw, dh = compute_i2v_output_dimensions_from_bytes(image_bytes, 640)
        return await run_image_to_video(
            image_bytes=image_bytes,
            prompt=prompt,
            negative_prompt=NEGATIVE_PET_DANCE,
            width=dw,
            height=dh,
            num_frames=num_frames or DANCE_SHORT_NUM_FRAMES,
            frame_rate=DANCE_SHORT_FRAME_RATE,
            num_inference_steps=num_steps or DANCE_SHORT_NUM_STEPS,
            guidance_scale=DANCE_SHORT_GUIDANCE_SCALE,
            seed=seed,
            condition_strength=DANCE_CONDITION_STRENGTH,
        )
