"""
Production orchestration: dog image + registered dance video → video with matched motion.

Two execution paths (explicit, configurable):

1) **Default (production-proven)**: LTX-2 / LTX-2.3 via `run_image_to_video` with
   `reference_video_path` (ComfyUI workflow when enabled). Motion comes from the actual
   reference video file injected into the graph — not text-only choreography.

2) **Optional SDXL pose frames** (`DANCE_USE_SDXL_POSE_PIPELINE=true` + valid ComfyUI JSON):
   Extract pose sequence → render OpenPose-style skeleton images → per-frame SDXL+ControlNet+IPAdapter
   (workflow must be exported from your ComfyUI) → ffmpeg mux to MP4.

**Identity**: IPAdapter (or image conditioning in the exported graph) preserves subject appearance
across frames better than text alone; LoRA is preferred when you have a *trained* subject adapter.
This service does not train LoRAs — it consumes existing ComfyUI graphs and optional IPAdapter nodes.

**GPU safety**: ComfyUI calls are serialized with an asyncio semaphore; heavy sync work runs in
`run_in_executor`. Optional `torch.cuda.empty_cache()` after batches when CUDA is available.

**Goals vs implementation** (강아지 자동 인식·사람↔강아지 대치·Wan 단독 등): see
`docs/DANCE_GOALS_AND_GAPS.md` — this service does not run pet detection or human swap;
reference video drives motion; MediaPipe pose is human-centric when using pose_sdxl.
"""

from __future__ import annotations

import asyncio
import logging
import tempfile
import uuid
from pathlib import Path
from typing import Literal

from app.core.config import get_settings
from app.schemas.dance_generation_schema import (
    CharacterPreparationResult,
    DanceGenerationInternalState,
    DanceGenerationJobConfig,
    PoseExtractionResult,
)
from app.schemas.dance_schema import MotionSequence
from app.services.dance_service import run_dance_generate
from app.services.pose_service import (
    extract_poses_from_video,
    get_pose_cache_path_for_key,
    normalize_motion,
    render_pose_frame_to_png_bytes,
    write_motion_sequence_cache,
)

logger = logging.getLogger(__name__)

_dance_comfy_sem: asyncio.Semaphore | None = None


def _get_dance_comfy_semaphore() -> asyncio.Semaphore:
    global _dance_comfy_sem
    if _dance_comfy_sem is None:
        limit = max(1, int(get_settings().gpu_semaphore_limit))
        _dance_comfy_sem = asyncio.Semaphore(limit)
    return _dance_comfy_sem


class DanceGenerationService:
    """
    High-level dance video generation with pose extraction, optional SDXL frame path,
    and fallback to LTX + reference video (ComfyUI or diffusers).
    """

    async def prepare_character(self, image_bytes: bytes) -> CharacterPreparationResult:
        """Validate image bytes; optionally upload to ComfyUI for downstream nodes."""
        if not image_bytes or len(image_bytes) < 32:
            raise ValueError("Invalid or empty image bytes")

        def _sync_validate() -> tuple[int, int, str]:
            from io import BytesIO

            from PIL import Image, ImageOps

            img = Image.open(BytesIO(image_bytes))
            img = ImageOps.exif_transpose(img)
            img = img.convert("RGB")
            return img.size[0], img.size[1], img.format or "PNG"

        loop = asyncio.get_event_loop()
        w, h, fmt = await loop.run_in_executor(None, _sync_validate)

        comfy_name: str | None = None
        settings = get_settings()
        if settings.comfyui_enabled:
            try:
                from app.services.comfyui_service import upload_image

                up = await upload_image(image_bytes, filename=f"dance_char_{uuid.uuid4().hex[:10]}.png")
                comfy_name = up.get("name")
            except Exception as e:
                logger.warning("ComfyUI character upload skipped: %s", e)

        return CharacterPreparationResult(
            ok=True,
            width=w,
            height=h,
            format=str(fmt),
            comfy_upload_name=comfy_name,
        )

    async def extract_pose_sequence(
        self,
        reference_video_path: Path,
        cache_key: str,
    ) -> PoseExtractionResult:
        """
        Extract normalized pose sequence from reference video; persist to pose cache JSON.
        """
        path = Path(reference_video_path)
        if not path.is_file():
            raise FileNotFoundError(f"Reference video not found: {path}")

        cached = get_pose_cache_path_for_key(cache_key)
        if cached.exists():
            try:
                data = cached.read_text(encoding="utf-8")
                from pydantic import TypeAdapter

                motion = TypeAdapter(MotionSequence).validate_json(data)
                if motion.frames:
                    logger.info("Pose cache hit: %s (%d frames)", cached, len(motion.frames))
                    return PoseExtractionResult(
                        cache_path=str(cached),
                        frame_count=len(motion.frames),
                        fps=float(motion.fps),
                        source_video_path=str(path.resolve()),
                    )
            except Exception as e:
                logger.warning("Stale pose cache, re-extracting: %s", e)

        loop = asyncio.get_event_loop()
        raw = await loop.run_in_executor(None, lambda: extract_poses_from_video(path, fps_out=None))
        if raw is None or not raw.frames:
            raise RuntimeError(
                "Pose extraction failed. Install mediapipe + opencv, or set POSE_LANDMARKER_MODEL_PATH."
            )
        normalized = normalize_motion(raw)
        out_path = write_motion_sequence_cache(cache_key, normalized)
        return PoseExtractionResult(
            cache_path=str(out_path),
            frame_count=len(normalized.frames),
            fps=float(normalized.fps),
            source_video_path=str(path.resolve()),
        )

    async def generate_frame_batch(
        self,
        motion: MotionSequence,
        character_comfy_image_name: str,
        start_frame: int,
        end_frame: int,
        seed: int,
        positive_prompt: str,
        negative_prompt: str,
    ) -> list[bytes]:
        """
        Generate PNG frame bytes for [start_frame, end_frame) using ComfyUI pose workflow.
        Requires `pipelines/dance/dog_pose_generation.json` (exported API format) with
        LoadImage slots for character + pose skeleton.
        """
        from app.services.comfyui_service import (
            inject_dance_pose_workflow_inputs,
            load_dance_workflow_template,
            run_workflow_and_get_image,
            upload_image,
        )

        settings = get_settings()
        if not settings.comfyui_enabled:
            raise RuntimeError("ComfyUI disabled (COMFYUI_ENABLED=false)")

        wf = load_dance_workflow_template("dog_pose_generation.json")
        frames_out: list[bytes] = []
        w = max(64, min(1024, motion.width))
        h = max(64, min(1024, motion.height))

        for idx in range(start_frame, end_frame):
            if idx >= len(motion.frames):
                break
            png = await asyncio.get_event_loop().run_in_executor(
                None,
                lambda i=idx: render_pose_frame_to_png_bytes(motion.frames[i], w, h),
            )
            pose_up = await upload_image(png, filename=f"pose_{uuid.uuid4().hex[:8]}_{idx:04d}.png")
            pose_name = pose_up.get("name", "pose.png")

            injected = inject_dance_pose_workflow_inputs(
                wf,
                character_image_name=character_comfy_image_name,
                pose_image_name=pose_name,
                seed=seed + idx,
                positive_prompt=positive_prompt,
                negative_prompt=negative_prompt,
            )
            async with _get_dance_comfy_semaphore():
                img_bytes = await run_workflow_and_get_image(
                    injected,
                    poll_interval=0.5,
                    max_wait=settings.comfyui_timeout_seconds,
                )
            frames_out.append(img_bytes)
            _maybe_empty_cuda_cache()

        return frames_out


    async def generate_video(self, frame_png_bytes: list[bytes], fps: float) -> bytes:
        """
        Encode ordered PNG frames to H.264 MP4 via ffmpeg (async via executor).
        """
        if not frame_png_bytes:
            raise ValueError("No frames to encode")

        def _ffmpeg_encode() -> bytes:
            import os
            import shutil
            import subprocess

            if shutil.which("ffmpeg") is None:
                raise RuntimeError("ffmpeg not found on PATH; install ffmpeg for frame→video muxing.")

            tmpdir = tempfile.mkdtemp(prefix="dance_frames_")
            try:
                for i, blob in enumerate(frame_png_bytes):
                    p = Path(tmpdir) / f"f_{i:06d}.png"
                    p.write_bytes(blob)
                out_mp4 = Path(tmpdir) / "out.mp4"
                cmd = [
                    "ffmpeg",
                    "-y",
                    "-framerate",
                    str(fps),
                    "-i",
                    str(Path(tmpdir) / "f_%06d.png"),
                    "-c:v",
                    "libx264",
                    "-pix_fmt",
                    "yuv420p",
                    "-movflags",
                    "+faststart",
                    str(out_mp4),
                ]
                subprocess.run(cmd, check=True, capture_output=True, text=True)
                return out_mp4.read_bytes()
            finally:
                for root, dirs, files in os.walk(tmpdir, topdown=False):
                    for name in files:
                        try:
                            os.unlink(Path(root) / name)
                        except OSError:
                            pass
                    try:
                        os.rmdir(root)
                    except OSError:
                        pass

        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(None, _ffmpeg_encode)

    async def _execute_pose_sdxl_pipeline(
        self,
        character: Literal["dog", "cat"],
        prep: CharacterPreparationResult,
        ref_path: Path,
        cache_key: str,
        cfg: DanceGenerationJobConfig,
    ) -> tuple[bytes, float, DanceGenerationInternalState]:
        """Pose extract → ComfyUI per-frame (dog_pose_generation.json) → ffmpeg MP4."""
        import time

        t0 = time.perf_counter()
        settings = get_settings()
        wf_path = settings.pipelines_dir / "dance" / "dog_pose_generation.json"
        if not wf_path.is_file():
            raise FileNotFoundError(f"Pose workflow not found: {wf_path}")
        if not prep.comfy_upload_name:
            raise RuntimeError(
                "ComfyUI에 캐릭터 이미지를 올릴 수 없어 pose_sdxl 파이프라인을 실행할 수 없습니다. "
                "COMFYUI_ENABLED=true 및 ComfyUI 서버를 확인하세요."
            )
        pose_res = await self.extract_pose_sequence(ref_path, cache_key=cache_key)
        from pydantic import TypeAdapter

        motion = TypeAdapter(MotionSequence).validate_json(
            Path(pose_res.cache_path).read_text(encoding="utf-8")
        )
        max_f = min(cfg.max_frames, len(motion.frames))
        frames_png: list[bytes] = []
        seed = cfg.seed if cfg.seed is not None else 42
        pos_prompt, neg_prompt = _dance_prompts(character)
        for start in range(0, max_f, cfg.batch_size):
            end = min(max_f, start + cfg.batch_size)
            batch = await self.generate_frame_batch(
                motion,
                prep.comfy_upload_name,
                start,
                end,
                seed=seed,
                positive_prompt=pos_prompt,
                negative_prompt=neg_prompt,
            )
            frames_png.extend(batch)
        vid = await self.generate_video(frames_png, fps=float(motion.fps))
        state = DanceGenerationInternalState(
            pipeline="comfyui_sdxl_pose_frames",
            pose_cache_path=pose_res.cache_path,
            frame_png_count=len(frames_png),
        )
        return vid, time.perf_counter() - t0, state

    async def generate_custom_dance_video(
        self,
        image_bytes: bytes,
        reference_video_bytes: bytes,
        character: Literal["dog", "cat"],
        reference_video_filename: str | None,
        job: DanceGenerationJobConfig,
    ) -> tuple[bytes, float, DanceGenerationInternalState]:
        """커스텀 레퍼런스 영상: pipeline=ltx → 기존 LTX 경로, pipeline=pose_sdxl → 포즈 프레임 경로."""
        from app.services.dance_service import run_dance_generate_custom

        settings = get_settings()
        cfg = job
        cfg.batch_size = settings.dance_frame_batch_size
        cfg.max_frames = settings.dance_max_pose_frames

        if cfg.pipeline == "ltx":
            out_bytes, elapsed = await run_dance_generate_custom(
                image_bytes=image_bytes,
                reference_video_bytes=reference_video_bytes,
                character=character,
                reference_video_filename=reference_video_filename,
            )
            return (
                out_bytes,
                elapsed,
                DanceGenerationInternalState(
                    pipeline="ltx_reference_video",
                    pose_cache_path=None,
                    frame_png_count=None,
                ),
            )

        motions_dir = settings.motions_dir
        motions_dir.mkdir(parents=True, exist_ok=True)
        temp_id = f"custom_{uuid.uuid4().hex[:8]}"
        allowed_suffixes = {".mp4", ".mov", ".webm", ".mkv", ".avi", ".m4v"}
        input_suffix = Path(reference_video_filename or "").suffix.lower()
        suffix = input_suffix if input_suffix in allowed_suffixes else ".mp4"
        temp_video_path = motions_dir / f"{temp_id}{suffix}"
        temp_video_path.write_bytes(reference_video_bytes)
        try:
            prep = await self.prepare_character(image_bytes)
            if not prep.ok:
                raise ValueError("Character preparation failed")
            wf_path = settings.pipelines_dir / "dance" / "dog_pose_generation.json"
            if wf_path.is_file() and prep.comfy_upload_name:
                try:
                    return await self._execute_pose_sdxl_pipeline(
                        character, prep, temp_video_path, temp_id, cfg
                    )
                except Exception as e:
                    logger.exception("custom pose_sdxl failed, falling back to LTX: %s", e)
            out_bytes, elapsed = await run_dance_generate_custom(
                image_bytes=image_bytes,
                reference_video_bytes=reference_video_bytes,
                character=character,
                reference_video_filename=reference_video_filename,
            )
            return (
                out_bytes,
                elapsed,
                DanceGenerationInternalState(
                    pipeline="ltx_reference_video",
                    pose_cache_path=None,
                    frame_png_count=None,
                ),
            )
        finally:
            try:
                if temp_video_path.exists():
                    temp_video_path.unlink()
            except OSError:
                pass

    async def generate_dance_video(
        self,
        image_bytes: bytes,
        motion_id: str,
        character: Literal["dog", "cat"] = "dog",
        reference_video_path: Path | None = None,
        job: DanceGenerationJobConfig | None = None,
    ) -> tuple[bytes, float, DanceGenerationInternalState]:
        """
        Main entry: character image + dance id (+ optional explicit video path) → MP4 bytes.

        - pipeline=ltx (default): LTX + reference video injection.
        - pipeline=pose_sdxl: pose extract → ComfyUI frames (pipelines/dance/dog_pose_generation.json) → ffmpeg.
        """
        settings = get_settings()
        cfg = job or DanceGenerationJobConfig(motion_id=motion_id, character=character)
        cfg.motion_id = motion_id
        cfg.character = character
        cfg.batch_size = settings.dance_frame_batch_size
        cfg.max_frames = settings.dance_max_pose_frames

        prep = await self.prepare_character(image_bytes)
        if not prep.ok:
            raise ValueError("Character preparation failed")

        wf_path = settings.pipelines_dir / "dance" / "dog_pose_generation.json"
        want_pose = cfg.pipeline == "pose_sdxl"
        if want_pose and wf_path.is_file() and prep.comfy_upload_name:
            try:
                ref_for_pose = reference_video_path or await self._resolve_reference_video_path(
                    motion_id, settings
                )
                return await self._execute_pose_sdxl_pipeline(
                    character, prep, ref_for_pose, motion_id, cfg
                )
            except Exception as e:
                logger.exception("pose_sdxl pipeline failed, falling back to LTX+reference: %s", e)

        # Fallback: proven path — LTX / ComfyUI LTX with reference video file
        ref = reference_video_path
        if ref is None:
            from app.services.dance_service import get_motion_video_path
            from app.services.dance_library import DanceLibrary

            lib = DanceLibrary(settings.dance_videos_dir)
            dance = await lib.get(motion_id)
            if dance is not None:
                ref = Path(dance.path)
            else:
                ref_path = get_motion_video_path(motion_id)
                if ref_path is None:
                    raise FileNotFoundError(
                        f"No reference video for motion_id={motion_id}. "
                        "Add a video under dance_videos/ or motions/ and refresh."
                    )
                ref = ref_path

        out_bytes, elapsed = await run_dance_generate(
            image_bytes=image_bytes,
            motion_id=motion_id,
            character=character,
            reference_video_path=ref,
        )
        state = DanceGenerationInternalState(
            pipeline="ltx_reference_video",
            pose_cache_path=str(get_pose_cache_path_for_key(motion_id)),
            frame_png_count=None,
        )
        return out_bytes, elapsed, state

    @staticmethod
    async def _resolve_reference_video_path(motion_id: str, settings) -> Path:
        from app.services.dance_service import get_motion_video_path
        from app.services.dance_library import DanceLibrary

        lib = DanceLibrary(settings.dance_videos_dir)
        dance = await lib.get(motion_id)
        if dance is not None:
            return Path(dance.path)
        ref_path = get_motion_video_path(motion_id)
        if ref_path is None:
            raise FileNotFoundError(
                f"No reference video for motion_id={motion_id}. "
                "Add a video under dance_videos/ or motions/ and refresh."
            )
        return ref_path


def _dance_prompts(character: str) -> tuple[str, str]:
    pos = (
        "full body cute dog, same identity as reference image, consistent fur markings, "
        "natural lighting, sharp focus, cinematic"
        if character == "dog"
        else "full body cute cat, same identity as reference image, consistent fur markings, "
        "natural lighting, sharp focus, cinematic"
    )
    neg = (
        "blurry, low quality, deformed, extra limbs, human body, text, watermark, "
        "identity drift, different animal"
    )
    return pos, neg


def _maybe_empty_cuda_cache() -> None:
    try:
        import torch

        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    except Exception:
        pass
