"""
Dance / Motion Transfer pipeline: pose extraction → normalization → video generation.

프롬프트 구조 (이미지 고정력 깨기):
  subject → clear action → repeated motion → rhythm → camera stability
  필수: "starts/begins" + "repeatedly" + "continuously" (again and again, continuous movement)
  "dance challenge"보다 "the dog starts dancing", "the dog moves" 같은 직접 행동 문장이 잘 먹힘.

파라미터: 640x384, 25 frames(8n+1), 8 steps, guidance 3, fps 8.
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

# image→video 모션 트리거: "상태"가 아니라 "행동"으로 써야 이미지 고정력이 깨짐.
# 구조: subject → clear action → repeated motion → rhythm → camera stability
# 필수: starts/begins + repeatedly + continuously (again and again, continuous movement)
DANCE_PROMPTS: dict[str, str] = {
    "rat_dance": (
        "a cute dog standing on the ground, full body visible. "
        "the dog begins a simple dance, "
        "moving its body left and right repeatedly, "
        "taking small steps in place again and again, "
        "its tail wagging while it dances. "
        "clear continuous movement, simple looping dance motion, static camera."
    ),
}

# 강아지: 행동 동사(begins) + repeatedly + again and again + continuous movement
PROMPT_DOG_DANCE = (
    "a cute dog standing on the ground, full body visible. "
    "the dog begins a simple dance, "
    "moving its body left and right repeatedly, "
    "taking small steps in place again and again, "
    "its tail wagging while it dances. "
    "clear continuous movement, simple looping dance motion, smooth natural dog motion, static camera."
)

# 고양이: 동일 구조 (행동 트리거)
PROMPT_CAT_DANCE = (
    "a cute cat standing on the ground, full body visible. "
    "the cat begins a simple dance, "
    "moving its body left and right repeatedly, "
    "taking small steps in place again and again, "
    "its tail swaying while it dances. "
    "clear continuous movement, simple looping dance motion, smooth natural cat motion, static camera."
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
    )
    return out_bytes, elapsed
