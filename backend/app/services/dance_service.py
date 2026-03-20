"""
Dance / Motion Transfer pipeline: pose extraction → normalization → video generation.

프롬프트: "dance" 대신 shift weight + very small steps. guidance 2.0~2.8, motion strength 0.35~0.50.
파라미터: 640x384, 33 frames(8n+1), 8 steps, guidance 2.3, fps 12, condition_strength 0.42.
"""

from __future__ import annotations

import asyncio
import json
import logging
from pathlib import Path

from app.core.config import get_settings
from app.schemas.dance_schema import MotionSequence
from app.services.pose_service import extract_poses_from_video, normalize_motion
from app.services.video_service import run_image_to_video

logger = logging.getLogger(__name__)

# Motion ID → reference video filename (under motions_dir)
MOTION_VIDEOS: dict[str, str] = {
    "rat_dance": "rat_dance.mp4",
}

# "dance" 단어는 LTX-2에서 큰 움직임으로 해석됨 → shift weight, small steps 사용. 동물+관절 조합에서 오류 많음.
# 작은 sway, 작은 걸음, 꼬리 wag 정도만. 구조 고정 문장 필수.
DANCE_PROMPTS: dict[str, str] = {
    "rat_dance": (
        "a cute dog standing on the ground, full body visible. "
        "the dog gently shifts its body weight left and right, making very small steps in place. "
        "its tail wagging slowly. "
        "slow natural dog movement, simple repeating motion. "
        "consistent dog appearance across frames, stable head and body. "
        "static camera, fixed framing."
    ),
}

# 강아지: dance ❌ / shift weight ✔ / small steps ✔
PROMPT_DOG_DANCE = (
    "a cute dog standing on the ground, full body visible. "
    "the dog gently shifts its body weight left and right, making very small steps in place. "
    "its tail wagging slowly. "
    "slow natural dog movement, simple repeating motion. "
    "consistent dog appearance across frames, stable head and body. "
    "static camera, fixed framing."
)

# 고양이: 동일 (shift weight, small steps, dance 단어 없음)
PROMPT_CAT_DANCE = (
    "a cute cat standing on the ground, full body visible. "
    "the cat gently shifts its body weight left and right, making very small steps in place. "
    "its tail swaying slowly. "
    "slow natural cat movement, simple repeating motion. "
    "consistent cat appearance across frames, stable head and body. "
    "static camera, fixed framing."
)

PROMPT_PET_GENERIC = PROMPT_DOG_DANCE
DEFAULT_DANCE_PROMPT = PROMPT_DOG_DANCE


def get_motion_video_path(motion_id: str) -> Path | None:
    """Return path to reference video for motion_id, or None if not found."""
    settings = get_settings()
    filename = MOTION_VIDEOS.get(motion_id)
    if not filename:
        return None
    path = settings.motions_dir / filename
    return path if path.exists() and path.is_file() else None


def _pose_cache_path(motion_id: str) -> Path:
    settings = get_settings()
    return settings.pose_cache_dir / f"{motion_id}.json"


def load_cached_pose(motion_id: str) -> MotionSequence | None:
    """Load normalized pose sequence from cache if present."""
    path = _pose_cache_path(motion_id)
    if not path.exists():
        return None
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
        return MotionSequence.model_validate(data)
    except Exception as e:
        logger.warning("Failed to load pose cache %s: %s", path, e)
        return None


def save_pose_cache(motion_id: str, motion: MotionSequence) -> None:
    """Save normalized motion to cache."""
    path = _pose_cache_path(motion_id)
    try:
        path.write_text(motion.model_dump_json(indent=0), encoding="utf-8")
        logger.info("Pose cache saved: %s (%d frames)", path, len(motion.frames))
    except Exception as e:
        logger.warning("Failed to save pose cache %s: %s", path, e)


def get_or_extract_pose(motion_id: str) -> MotionSequence | None:
    """
    Get normalized pose sequence for motion_id: from cache or by extracting from reference video.
    Returns None if no video or extraction fails.
    """
    cached = load_cached_pose(motion_id)
    if cached is not None:
        return cached
    video_path = get_motion_video_path(motion_id)
    if video_path is None:
        logger.warning("No reference video for motion_id=%s", motion_id)
        return None
    return get_or_extract_pose_from_path(video_path, cache_key=motion_id)


def get_or_extract_pose_from_path(
    video_path: Path,
    cache_key: str | None = None,
    fps_out: float = 30.0,
) -> MotionSequence | None:
    """
    Extract and normalize pose from a video file. Optionally cache by cache_key (e.g. motion_id).
    Returns None if extraction fails or empty.
    """
    if cache_key:
        cached = load_cached_pose(cache_key)
        if cached is not None:
            return cached
    if not video_path.exists() or not video_path.is_file():
        return None
    raw = extract_poses_from_video(video_path, fps_out=fps_out)
    if raw is None or not raw.frames:
        logger.warning("Pose extraction failed or empty for path=%s", video_path)
        return None
    normalized = normalize_motion(raw)
    if cache_key:
        save_pose_cache(cache_key, normalized)
    return normalized


def get_dance_prompt(motion_id: str, character: str) -> str:
    """Return LTX-2 prompt for the given motion and character (반려동물 관절에 맞는 자연스러운 춤)."""
    if character == "cat":
        return PROMPT_CAT_DANCE
    if character == "dog":
        return DANCE_PROMPTS.get(motion_id) or PROMPT_DOG_DANCE
    return PROMPT_PET_GENERIC


async def run_dance_generate_custom(
    image_bytes: bytes,
    reference_video_bytes: bytes,
    character: str = "dog",
    reference_video_filename: str | None = None,
) -> tuple[bytes, float]:
    """
    Run dance video generation from a user-uploaded reference video.
    Saves the reference video temporarily, extracts poses, then generates video.
    Returns (video_bytes, processing_time_seconds).
    """
    import tempfile
    import os
    settings = get_settings()
    motions_dir = settings.motions_dir
    motions_dir.mkdir(parents=True, exist_ok=True)

    # 임시 motion ID로 레퍼런스 영상 저장 (원본 확장자 유지: mov/webm/mp4 등)
    import uuid
    temp_motion_id = f"custom_{uuid.uuid4().hex[:8]}"
    allowed_suffixes = {".mp4", ".mov", ".webm", ".mkv", ".avi", ".m4v"}
    input_suffix = Path(reference_video_filename or "").suffix.lower()
    suffix = input_suffix if input_suffix in allowed_suffixes else ".mp4"
    temp_video_path = motions_dir / f"{temp_motion_id}{suffix}"

    try:
        temp_video_path.write_bytes(reference_video_bytes)
        logger.info("Custom reference video saved: %s", temp_video_path)

        # MOTION_VIDEOS에 임시 등록 (포즈 추출 시 사용)
        MOTION_VIDEOS[temp_motion_id] = f"{temp_motion_id}{suffix}"

        # 포즈 추출 시도 (선택). 실패해도 레퍼런스 영상 파일을 ComfyUI에 넘겨 동작 반영 가능.
        loop = asyncio.get_event_loop()
        await loop.run_in_executor(None, get_or_extract_pose, temp_motion_id)
    finally:
        MOTION_VIDEOS.pop(temp_motion_id, None)

    from app.services.video_service import (
        DANCE_CONDITION_STRENGTH,
        DANCE_SHORT_FRAME_RATE,
        DANCE_SHORT_GUIDANCE_SCALE,
        DANCE_SHORT_HEIGHT,
        DANCE_SHORT_NUM_FRAMES,
        DANCE_SHORT_NUM_STEPS,
        DANCE_SHORT_WIDTH,
        NEGATIVE_PET_DANCE,
    )

    # 레퍼런스 영상 경로를 비디오 생성에 전달. ComfyUI 워크플로에 레퍼런스 입력 노드가 있고
    # COMFYUI_REFERENCE_VIDEO_DIR 이 설정되어 있으면 동일 동작으로 생성됨.
    prompt = get_dance_prompt("rat_dance", character)
    try:
        out_bytes, elapsed = await run_image_to_video(
            image_bytes=image_bytes,
            prompt=prompt,
            negative_prompt=NEGATIVE_PET_DANCE,
            width=DANCE_SHORT_WIDTH,
            height=DANCE_SHORT_HEIGHT,
            num_frames=DANCE_SHORT_NUM_FRAMES,
            frame_rate=DANCE_SHORT_FRAME_RATE,
            num_inference_steps=DANCE_SHORT_NUM_STEPS,
            guidance_scale=DANCE_SHORT_GUIDANCE_SCALE,
            seed=None,
            condition_strength=DANCE_CONDITION_STRENGTH,
            reference_video_path=temp_video_path,
        )
        return out_bytes, elapsed
    finally:
        try:
            if temp_video_path.exists():
                temp_video_path.unlink()
        except Exception:
            pass


async def run_dance_generate(
    image_bytes: bytes,
    motion_id: str,
    character: str = "dog",
    reference_video_path: Path | None = None,
) -> tuple[bytes, float]:
    """
    Run dance video generation: optionally ensure pose cache, then run LTX-2 image-to-video
    with the character image and dance prompt.
    reference_video_path 가 있으면 해당 경로를 레퍼런스 영상으로 사용(라이브러리 등에서 전달).
    Returns (video_bytes, processing_time_seconds).
    """
    from app.services.video_service import (
        DANCE_CONDITION_STRENGTH,
        DANCE_SHORT_FRAME_RATE,
        DANCE_SHORT_GUIDANCE_SCALE,
        DANCE_SHORT_HEIGHT,
        DANCE_SHORT_NUM_FRAMES,
        DANCE_SHORT_NUM_STEPS,
        DANCE_SHORT_WIDTH,
        NEGATIVE_PET_DANCE,
    )

    video_path = reference_video_path or get_motion_video_path(motion_id)
    if video_path is not None:
        loop = asyncio.get_event_loop()
        await loop.run_in_executor(
            None,
            lambda: get_or_extract_pose_from_path(video_path, cache_key=motion_id),
        )

    prompt = get_dance_prompt(motion_id, character)
    out_bytes, elapsed = await run_image_to_video(
        image_bytes=image_bytes,
        prompt=prompt,
        negative_prompt=NEGATIVE_PET_DANCE,
        width=DANCE_SHORT_WIDTH,
        height=DANCE_SHORT_HEIGHT,
        num_frames=DANCE_SHORT_NUM_FRAMES,
        frame_rate=DANCE_SHORT_FRAME_RATE,
        num_inference_steps=DANCE_SHORT_NUM_STEPS,
        guidance_scale=DANCE_SHORT_GUIDANCE_SCALE,
        seed=None,
        condition_strength=DANCE_CONDITION_STRENGTH,
        reference_video_path=video_path,
    )
    return out_bytes, elapsed
