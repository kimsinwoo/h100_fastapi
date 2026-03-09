"""
Dance / Motion Transfer pipeline: pose extraction → normalization → video generation.
Orchestrates: reference video → pose extraction (OpenPose-style) → motion normalization
→ LTX-2 image-to-video with dance prompt (pose conditioning can be added later via ControlNet).
"""

from __future__ import annotations

import asyncio
import json
import logging
import time
from pathlib import Path
from typing import Any

from app.core.config import get_settings
from app.schemas.dance_schema import MotionSequence
from app.services.pose_service import extract_poses_from_video, normalize_motion
from app.services.video_service import run_image_to_video

logger = logging.getLogger(__name__)

# Motion ID → reference video filename (under motions_dir)
MOTION_VIDEOS: dict[str, str] = {
    "rat_dance": "rat_dance.mp4",
}

# Dance prompts per motion_id (LTX-2; pose conditioning TBD with AnimateDiff+ControlNet)
DANCE_PROMPTS: dict[str, str] = {
    "rat_dance": (
        "A fixed camera medium shot of a cute dog standing on its hind legs in the center of the frame. "
        "The dog performs the RAT Dance Challenge: it suddenly begins dancing energetically, swaying body left and right, "
        "raising and waving its front paws in rhythm. It does small rhythmic hops on hind legs, bouncing lightly, "
        "shifting weight side to side. Tail wags happily, ears bounce with each move, head tilts playfully. "
        "The dog continues with repeated paw waves, little jumps, and lively body sways. Camera and background stay still."
    ),
}

DEFAULT_DANCE_PROMPT = (
    "A fixed camera medium shot of a cute dog dancing on its hind legs in the center of the frame. "
    "The pet begins dancing energetically, swaying and waving its front paws, bouncing with small hops. "
    "Tail wags, ears bounce, head tilts playfully. Continuous motion; camera and background remain still."
)


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
    """Return LTX-2 prompt for the given motion and character."""
    prompt = DANCE_PROMPTS.get(motion_id) or DEFAULT_DANCE_PROMPT
    if character == "cat":
        prompt = prompt.replace("dog", "cat").replace("Dog", "Cat")
    return prompt


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
        DEFAULT_HEIGHT,
        DEFAULT_NUM_FRAMES,
        DEFAULT_WIDTH,
    )

    prompt = get_dance_prompt(motion_id, character)
    negative = None
    start = time.perf_counter()
    out_bytes, elapsed = await run_image_to_video(
        image_bytes=image_bytes,
        prompt=prompt,
        negative_prompt=negative,
        width=DEFAULT_WIDTH,
        height=DEFAULT_HEIGHT,
        num_frames=DEFAULT_NUM_FRAMES,
        frame_rate=30.0,
        num_inference_steps=25,
        guidance_scale=4.0,
        seed=None,
    )
    return out_bytes, elapsed
