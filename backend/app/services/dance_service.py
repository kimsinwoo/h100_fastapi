"""
Dance / Motion Transfer pipeline: pose extraction → normalization → video generation.

프롬프트: slow start + small subtle sway + tiny steps + consistent structure across frames, stable head and face.
울렁임 방지: 구조 고정 문장, small/subtle sway, 해상도 512~768, frames 33~49.
파라미터: 640x384, 49 frames(8n+1), 8 steps, guidance 2.5~3.5, fps 8, condition_strength 0.6.
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

# 울렁임/깨짐 방지: 1) 프레임 간 구조 유지 2) small subtle sway 3) slow start
# 구조 고정: consistent body structure across frames, stable head and face
# sway: small subtle body sway (큰 sway는 몸 회전으로 해석되어 찌그러짐)
# 초기 프레임: the motion starts slowly
DANCE_PROMPTS: dict[str, str] = {
    "rat_dance": (
        "a cute dog standing on the ground, full body visible. "
        "the dog slowly starts a simple dance, "
        "making a small subtle body sway left and right, taking tiny steps in place. "
        "its tail wagging gently. "
        "the movement is slow and repeating in a simple rhythm. "
        "consistent dog appearance across frames, consistent body structure, stable head and face. "
        "static camera, fixed framing."
    ),
}

# 강아지: slow start + small subtle sway + tiny steps + 구조 고정 문장
PROMPT_DOG_DANCE = (
    "a cute dog standing on the ground, full body visible. "
    "the dog slowly starts a simple dance, "
    "making a small subtle body sway left and right, taking tiny steps in place. "
    "its tail wagging gently. "
    "the movement is slow and repeating in a simple rhythm. "
    "consistent dog appearance across frames, consistent body structure, stable head and face. "
    "static camera, fixed framing."
)

# 고양이: 동일 (slow start, small subtle sway, 구조 고정)
PROMPT_CAT_DANCE = (
    "a cute cat standing on the ground, full body visible. "
    "the cat slowly starts a simple dance, "
    "making a small subtle body sway left and right, taking tiny steps in place. "
    "its tail swaying gently. "
    "the movement is slow and repeating in a simple rhythm. "
    "consistent cat appearance across frames, consistent body structure, stable head and face. "
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
    raw = extract_poses_from_video(video_path, fps_out=30.0)
    if raw is None or not raw.frames:
        logger.warning("Pose extraction failed or empty for motion_id=%s", motion_id)
        return None
    normalized = normalize_motion(raw)
    save_pose_cache(motion_id, normalized)
    return normalized


def get_dance_prompt(motion_id: str, character: str) -> str:
    """Return LTX-2 prompt for the given motion and character (반려동물 관절에 맞는 자연스러운 춤)."""
    if character == "cat":
        return PROMPT_CAT_DANCE
    if character == "dog":
        return DANCE_PROMPTS.get(motion_id) or PROMPT_DOG_DANCE
    return PROMPT_PET_GENERIC


async def run_dance_generate(
    image_bytes: bytes,
    motion_id: str,
    character: str = "dog",
) -> tuple[bytes, float]:
    """
    Run dance video generation: optionally ensure pose cache, then run LTX-2 image-to-video
    with the character image and dance prompt. Returns (video_bytes, processing_time_seconds).
    """
    # Pre-warm pose cache in background (optional)
    loop = asyncio.get_event_loop()
    await loop.run_in_executor(None, get_or_extract_pose, motion_id)

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
    )
    return out_bytes, elapsed
