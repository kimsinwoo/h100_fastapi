"""
Dataset preparation pipeline for Reels dance motion LoRA training.

Responsibilities:
  - Load dance videos (mp4 / mov) from a category directory
  - Extract frames at a target FPS using decord (fast) or cv2 (fallback)
  - Extract DW Pose / OpenPose skeleton sequences per frame
  - Render pose skeleton images for pose-conditioning
  - Cache processed data (pickle) to avoid re-processing on every run
  - Export a flat list of VideoClip objects ready for motion_dataset.py
"""

from __future__ import annotations

import hashlib
import logging
import os
import pickle
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Optional, Tuple

import cv2
import numpy as np

logger = logging.getLogger(__name__)

# ──────────────────────────────────────────────────────────────────────────────
# Constants
# ──────────────────────────────────────────────────────────────────────────────
SUPPORTED_VIDEO_EXTS = {".mp4", ".mov", ".avi", ".webm", ".mkv"}
DEFAULT_FPS_OUT = 8.0          # sampling rate for training clips
DEFAULT_CLIP_FRAMES = 16       # frames per training clip
DEFAULT_TARGET_SIZE = (512, 512)  # (W, H) resize for training

# COCO-18 body keypoint connections for skeleton rendering
COCO_SKELETON_PAIRS = [
    (0, 1), (0, 2), (1, 3), (2, 4),
    (5, 6), (5, 7), (7, 9), (6, 8), (8, 10),
    (5, 11), (6, 12), (11, 12),
    (11, 13), (13, 15), (12, 14), (14, 16),
]
KEYPOINT_COLORS = [
    (255, 0, 0), (255, 85, 0), (255, 170, 0), (255, 255, 0),
    (170, 255, 0), (85, 255, 0), (0, 255, 0), (0, 255, 85),
    (0, 255, 170), (0, 255, 255), (0, 170, 255), (0, 85, 255),
    (0, 0, 255), (85, 0, 255), (170, 0, 255), (255, 0, 255),
    (255, 0, 170), (255, 0, 85),
]

# ──────────────────────────────────────────────────────────────────────────────
# Data structures
# ──────────────────────────────────────────────────────────────────────────────

@dataclass
class PoseKeypoints:
    """Single-frame pose keypoints (COCO-18 body + optional hands/face)."""
    body: np.ndarray       # (18, 3)  [x, y, score]  — normalized 0-1
    hand_left: Optional[np.ndarray] = None   # (21, 3)
    hand_right: Optional[np.ndarray] = None  # (21, 3)
    face: Optional[np.ndarray] = None        # (68, 3)


@dataclass
class FrameData:
    """One extracted video frame with optional pose annotation."""
    frame_idx: int
    timestamp: float            # seconds
    image: np.ndarray           # (H, W, 3) BGR uint8
    pose: Optional[PoseKeypoints] = None
    pose_image: Optional[np.ndarray] = None  # rendered skeleton (H, W, 3)


@dataclass
class VideoClip:
    """Processed video clip: sequential frames + metadata."""
    video_path: Path
    category: str               # e.g. "tiktok_shuffle"
    clip_index: int             # which clip within the video
    frames: List[FrameData]
    fps: float
    width: int
    height: int
    total_duration: float
    metadata: dict = field(default_factory=dict)


# ──────────────────────────────────────────────────────────────────────────────
# Pose extractor
# ──────────────────────────────────────────────────────────────────────────────

class DWPoseExtractor:
    """
    DW Pose / OpenPose skeleton extractor.

    Priority order:
      1. dwpose (mmpose-based, highest quality)
      2. mediapipe BlazePose (already in requirements, good quality)
      3. OpenCV DNN with OpenPose proto (fallback)
    """

    def __init__(self, device: str = "cuda"):
        self.device = device
        self._backend: str = "none"
        self._pose_fn = None
        self._init_backend()

    # ──────────────────────────────────────────────────────────────────────────
    def _init_backend(self) -> None:
        # 1) Try DW Pose (mmpose)
        try:
            from mmpose.apis import init_model, inference_topdown  # noqa: F401
            self._backend = "dwpose"
            self._init_dwpose()
            logger.info("[PoseExtractor] backend=dwpose (mmpose)")
            return
        except ImportError:
            pass

        # 2) Try MediaPipe BlazePose
        try:
            import mediapipe as mp  # noqa: F401
            self._backend = "mediapipe"
            self._init_mediapipe()
            logger.info("[PoseExtractor] backend=mediapipe")
            return
        except ImportError:
            pass

        # 3) Pure-CV2 fallback (no keypoints, zero array)
        self._backend = "cv2_noop"
        logger.warning(
            "[PoseExtractor] No pose backend found. "
            "Install mediapipe: pip install mediapipe>=0.10.0 "
            "or mmpose for DW Pose. Returning zero keypoints."
        )

    # ── DW Pose ───────────────────────────────────────────────────────────────
    def _init_dwpose(self) -> None:
        from mmpose.apis import init_model
        # DW Pose config + checkpoint paths (download separately)
        cfg = os.environ.get(
            "DWPOSE_CONFIG",
            "configs/dwpose/dwpose_l_384x288.py",
        )
        ckpt = os.environ.get(
            "DWPOSE_CHECKPOINT",
            "checkpoints/dw-ll_ucoco_384.pth",
        )
        try:
            self._dw_model = init_model(cfg, ckpt, device=self.device)
        except Exception as e:
            raise RuntimeError(f"DW Pose init failed: {e}") from e

    def _extract_dwpose(self, bgr: np.ndarray) -> PoseKeypoints:
        from mmpose.apis import inference_topdown
        from mmpose.utils import adapt_mmdet_pipeline

        h, w = bgr.shape[:2]
        results = inference_topdown(
            self._dw_model,
            bgr,
            bboxes=np.array([[0, 0, w, h, 1.0]]),
            bbox_format="xyxy",
        )
        if not results:
            return self._zero_keypoints()

        kps = results[0].pred_instances.keypoints[0]  # (133, 3) whole-body
        body = kps[:18].copy()
        # normalize to 0-1
        body[:, 0] /= w
        body[:, 1] /= h
        hand_left = kps[91:112].copy() if kps.shape[0] > 112 else None
        hand_right = kps[112:133].copy() if kps.shape[0] > 133 else None
        if hand_left is not None:
            hand_left[:, 0] /= w; hand_left[:, 1] /= h
        if hand_right is not None:
            hand_right[:, 0] /= w; hand_right[:, 1] /= h
        return PoseKeypoints(body=body, hand_left=hand_left, hand_right=hand_right)

    # ── MediaPipe ─────────────────────────────────────────────────────────────
    def _init_mediapipe(self) -> None:
        import mediapipe as mp
        self._mp_pose = mp.solutions.pose.Pose(
            static_image_mode=False,
            model_complexity=2,
            enable_segmentation=False,
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5,
        )
        # MediaPipe → COCO-18 index mapping
        # fmt: off
        self._mp_to_coco = [
            0,   # nose
            2,   # left eye
            5,   # right eye
            7,   # left ear
            8,   # right ear
            11,  # left shoulder
            12,  # right shoulder
            13,  # left elbow
            14,  # right elbow
            15,  # left wrist
            16,  # right wrist
            23,  # left hip
            24,  # right hip
            25,  # left knee
            26,  # right knee
            27,  # left ankle
            28,  # right ankle
            0,   # neck (approx nose)
        ]
        # fmt: on

    def _extract_mediapipe(self, bgr: np.ndarray) -> PoseKeypoints:
        import mediapipe as mp
        rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
        results = self._mp_pose.process(rgb)
        if not results.pose_landmarks:
            return self._zero_keypoints()

        lms = results.pose_landmarks.landmark
        body = np.zeros((18, 3), dtype=np.float32)
        for coco_idx, mp_idx in enumerate(self._mp_to_coco):
            lm = lms[mp_idx]
            body[coco_idx] = [lm.x, lm.y, lm.visibility]
        return PoseKeypoints(body=body)

    # ── Zero fallback ─────────────────────────────────────────────────────────
    @staticmethod
    def _zero_keypoints() -> PoseKeypoints:
        return PoseKeypoints(body=np.zeros((18, 3), dtype=np.float32))

    # ── Public API ────────────────────────────────────────────────────────────
    def extract(self, bgr: np.ndarray) -> PoseKeypoints:
        """Extract keypoints from a BGR frame."""
        if self._backend == "dwpose":
            return self._extract_dwpose(bgr)
        if self._backend == "mediapipe":
            return self._extract_mediapipe(bgr)
        return self._zero_keypoints()

    def close(self) -> None:
        if self._backend == "mediapipe" and self._mp_pose is not None:
            self._mp_pose.close()


# ──────────────────────────────────────────────────────────────────────────────
# Skeleton renderer
# ──────────────────────────────────────────────────────────────────────────────

def render_skeleton(
    pose: PoseKeypoints,
    width: int,
    height: int,
    thickness: int = 3,
    radius: int = 5,
) -> np.ndarray:
    """
    Render COCO-18 pose skeleton on a black canvas.
    Returns BGR uint8 image (height, width, 3).
    """
    canvas = np.zeros((height, width, 3), dtype=np.uint8)
    kps = pose.body  # (18, 3) normalized

    # Draw limb connections
    for (i, j) in COCO_SKELETON_PAIRS:
        if kps[i, 2] < 0.1 or kps[j, 2] < 0.1:
            continue
        pt1 = (int(kps[i, 0] * width), int(kps[i, 1] * height))
        pt2 = (int(kps[j, 0] * width), int(kps[j, 1] * height))
        color = KEYPOINT_COLORS[i % len(KEYPOINT_COLORS)]
        cv2.line(canvas, pt1, pt2, color, thickness, cv2.LINE_AA)

    # Draw keypoint dots
    for idx in range(min(18, len(kps))):
        if kps[idx, 2] < 0.1:
            continue
        cx = int(kps[idx, 0] * width)
        cy = int(kps[idx, 1] * height)
        cv2.circle(canvas, (cx, cy), radius, KEYPOINT_COLORS[idx % len(KEYPOINT_COLORS)], -1, cv2.LINE_AA)

    return canvas


# ──────────────────────────────────────────────────────────────────────────────
# Frame reader (decord-first, cv2 fallback)
# ──────────────────────────────────────────────────────────────────────────────

def _read_frames_decord(
    video_path: Path,
    target_fps: float,
    target_size: Tuple[int, int],
) -> Tuple[List[np.ndarray], float, float]:
    """
    Returns (frames_bgr, source_fps, duration).
    frames_bgr: list of (H, W, 3) uint8 BGR arrays at target_fps.
    """
    import decord
    decord.bridge.set_bridge("numpy")
    vr = decord.VideoReader(str(video_path))
    src_fps = float(vr.get_avg_fps())
    total = len(vr)
    duration = total / src_fps if src_fps > 0 else 0.0

    stride = max(1, round(src_fps / target_fps))
    indices = list(range(0, total, stride))
    raw = vr.get_batch(indices).asnumpy()  # (N, H, W, 3) RGB

    W, H = target_size
    frames = []
    for f in raw:
        f_bgr = cv2.cvtColor(f, cv2.COLOR_RGB2BGR)
        f_bgr = cv2.resize(f_bgr, (W, H), interpolation=cv2.INTER_AREA)
        frames.append(f_bgr)
    return frames, src_fps, duration


def _read_frames_cv2(
    video_path: Path,
    target_fps: float,
    target_size: Tuple[int, int],
) -> Tuple[List[np.ndarray], float, float]:
    cap = cv2.VideoCapture(str(video_path))
    src_fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    duration = total / src_fps

    stride = max(1, round(src_fps / target_fps))
    W, H = target_size
    frames = []
    idx = 0
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        if idx % stride == 0:
            frame = cv2.resize(frame, (W, H), interpolation=cv2.INTER_AREA)
            frames.append(frame)
        idx += 1
    cap.release()
    return frames, src_fps, duration


def read_video_frames(
    video_path: Path,
    target_fps: float = DEFAULT_FPS_OUT,
    target_size: Tuple[int, int] = DEFAULT_TARGET_SIZE,
) -> Tuple[List[np.ndarray], float, float]:
    """Read frames from a video file. Returns (frames_bgr, src_fps, duration)."""
    try:
        return _read_frames_decord(video_path, target_fps, target_size)
    except ImportError:
        logger.debug("decord not installed — falling back to cv2")
    except Exception as e:
        logger.warning("decord read failed for %s: %s — falling back to cv2", video_path, e)
    return _read_frames_cv2(video_path, target_fps, target_size)


# ──────────────────────────────────────────────────────────────────────────────
# Cache helpers
# ──────────────────────────────────────────────────────────────────────────────

def _cache_key(video_path: Path, target_fps: float, target_size: Tuple[int, int]) -> str:
    stat = video_path.stat()
    raw = f"{video_path.resolve()}{stat.st_size}{stat.st_mtime}{target_fps}{target_size}"
    return hashlib.md5(raw.encode()).hexdigest()


def _cache_path(cache_dir: Path, key: str) -> Path:
    return cache_dir / f"{key}.pkl"


def load_cached_clip(cache_dir: Path, key: str) -> Optional[List[FrameData]]:
    p = _cache_path(cache_dir, key)
    if not p.exists():
        return None
    try:
        with open(p, "rb") as f:
            return pickle.load(f)
    except Exception as e:
        logger.warning("Cache load failed (%s): %s", p, e)
        return None


def save_cached_clip(cache_dir: Path, key: str, frames: List[FrameData]) -> None:
    cache_dir.mkdir(parents=True, exist_ok=True)
    p = _cache_path(cache_dir, key)
    try:
        with open(p, "wb") as f:
            pickle.dump(frames, f, protocol=pickle.HIGHEST_PROTOCOL)
    except Exception as e:
        logger.warning("Cache save failed (%s): %s", p, e)


# ──────────────────────────────────────────────────────────────────────────────
# Main pipeline
# ──────────────────────────────────────────────────────────────────────────────

class ReelsDatasetPipeline:
    """
    Full dataset preparation pipeline.

    Usage:
        pipeline = ReelsDatasetPipeline(cache_dir=Path("data/reels_cache"))
        clips = pipeline.process_category(
            video_dir=Path("data/reels_raw/tiktok_shuffle"),
            category="tiktok_shuffle",
        )
    """

    def __init__(
        self,
        cache_dir: Path,
        target_fps: float = DEFAULT_FPS_OUT,
        target_size: Tuple[int, int] = DEFAULT_TARGET_SIZE,
        clip_frames: int = DEFAULT_CLIP_FRAMES,
        clip_overlap: int = 4,
        device: str = "cuda",
        skip_pose: bool = False,
    ):
        self.cache_dir = Path(cache_dir)
        self.target_fps = target_fps
        self.target_size = target_size
        self.clip_frames = clip_frames
        self.clip_overlap = clip_overlap
        self.skip_pose = skip_pose

        self._pose_extractor: Optional[DWPoseExtractor] = None
        if not skip_pose:
            self._pose_extractor = DWPoseExtractor(device=device)

    # ──────────────────────────────────────────────────────────────────────────
    def process_category(
        self,
        video_dir: Path,
        category: str,
        max_videos: Optional[int] = None,
    ) -> List[VideoClip]:
        """
        Process all videos in a directory for a given category.
        Returns a list of VideoClip objects.
        """
        video_dir = Path(video_dir)
        video_files = sorted(
            p for p in video_dir.iterdir()
            if p.suffix.lower() in SUPPORTED_VIDEO_EXTS
        )
        if max_videos:
            video_files = video_files[:max_videos]

        if not video_files:
            logger.warning("[Pipeline] No videos found in %s", video_dir)
            return []

        all_clips: List[VideoClip] = []
        for vf in video_files:
            try:
                clips = self.process_video(vf, category)
                all_clips.extend(clips)
                logger.info(
                    "[Pipeline] %s → %d clips (total %d)",
                    vf.name, len(clips), len(all_clips),
                )
            except Exception as e:
                logger.error("[Pipeline] Failed to process %s: %s", vf, e, exc_info=True)

        logger.info(
            "[Pipeline] Category '%s': %d videos → %d clips",
            category, len(video_files), len(all_clips),
        )
        return all_clips

    # ──────────────────────────────────────────────────────────────────────────
    def process_video(self, video_path: Path, category: str) -> List[VideoClip]:
        """
        Process a single video into a list of fixed-length VideoClips.
        Uses cache if available.
        """
        W, H = self.target_size
        cache_key = _cache_key(video_path, self.target_fps, self.target_size)
        cached_frames = load_cached_clip(self.cache_dir / category, cache_key)

        if cached_frames is not None:
            logger.debug("[Pipeline] Cache hit for %s", video_path.name)
            frames = cached_frames
            src_fps = self.target_fps
            duration = len(frames) / self.target_fps
        else:
            t0 = time.perf_counter()
            raw_frames, src_fps, duration = read_video_frames(
                video_path, self.target_fps, self.target_size
            )
            frames = self._build_frame_data(raw_frames, src_fps)
            save_cached_clip(self.cache_dir / category, cache_key, frames)
            logger.debug(
                "[Pipeline] Processed %s: %d frames in %.1fs",
                video_path.name, len(frames), time.perf_counter() - t0,
            )

        return self._split_into_clips(frames, video_path, category, src_fps, duration, W, H)

    # ──────────────────────────────────────────────────────────────────────────
    def _build_frame_data(
        self,
        raw_frames: List[np.ndarray],
        src_fps: float,
    ) -> List[FrameData]:
        """Extract pose for each frame and return FrameData list."""
        frames: List[FrameData] = []
        W, H = self.target_size

        for i, bgr in enumerate(raw_frames):
            ts = i / self.target_fps
            if self._pose_extractor is not None:
                pose = self._pose_extractor.extract(bgr)
                pose_img = render_skeleton(pose, W, H)
            else:
                pose = None
                pose_img = None

            frames.append(FrameData(
                frame_idx=i,
                timestamp=ts,
                image=bgr,
                pose=pose,
                pose_image=pose_img,
            ))
        return frames

    # ──────────────────────────────────────────────────────────────────────────
    def _split_into_clips(
        self,
        frames: List[FrameData],
        video_path: Path,
        category: str,
        src_fps: float,
        duration: float,
        width: int,
        height: int,
    ) -> List[VideoClip]:
        """Split frame list into overlapping fixed-length clips."""
        step = self.clip_frames - self.clip_overlap
        clips: List[VideoClip] = []
        clip_idx = 0

        for start in range(0, len(frames) - self.clip_frames + 1, step):
            clip_frames = frames[start: start + self.clip_frames]
            clips.append(VideoClip(
                video_path=video_path,
                category=category,
                clip_index=clip_idx,
                frames=clip_frames,
                fps=self.target_fps,
                width=width,
                height=height,
                total_duration=duration,
                metadata={"source_fps": src_fps},
            ))
            clip_idx += 1

        return clips

    # ──────────────────────────────────────────────────────────────────────────
    def close(self) -> None:
        if self._pose_extractor is not None:
            self._pose_extractor.close()

    def __enter__(self) -> "ReelsDatasetPipeline":
        return self

    def __exit__(self, *_) -> None:
        self.close()
