"""
Pose extraction from video (OpenPose-style keypoints) and motion normalization.
Uses MediaPipe Pose for body skeleton; output format compatible with motion conditioning.
"""

from __future__ import annotations

import logging
from pathlib import Path

from app.schemas.dance_schema import FramePose, Keypoint, MotionSequence

logger = logging.getLogger(__name__)

# MediaPipe Pose landmark index -> joint name (OpenPose-style)
MEDIAPIPE_TO_JOINT: list[tuple[int, str]] = [
    (0, "nose"),
    (1, "left_eye_inner"),
    (2, "left_eye"),
    (3, "left_eye_outer"),
    (4, "right_eye_inner"),
    (5, "right_eye"),
    (6, "right_eye_outer"),
    (7, "left_ear"),
    (8, "right_ear"),
    (9, "mouth_left"),
    (10, "mouth_right"),
    (11, "left_shoulder"),
    (12, "right_shoulder"),
    (13, "left_elbow"),
    (14, "right_elbow"),
    (15, "left_wrist"),
    (16, "right_wrist"),
    (17, "left_pinky"),
    (18, "right_pinky"),
    (19, "left_index"),
    (20, "right_index"),
    (21, "left_thumb"),
    (22, "right_thumb"),
    (23, "left_hip"),
    (24, "right_hip"),
    (25, "left_knee"),
    (26, "right_knee"),
    (27, "left_ankle"),
    (28, "right_ankle"),
    (29, "left_heel"),
    (30, "right_heel"),
    (31, "left_foot_index"),
    (32, "right_foot_index"),
]


def _extract_poses_mediapipe(video_path: Path, fps_out: float | None = None) -> MotionSequence | None:
    """Extract body keypoints per frame using MediaPipe Pose. Returns None if dependency missing."""
    try:
        import cv2
    except ImportError as e:
        logger.warning("Pose extraction requires opencv-python: %s", e)
        return None

    try:
        # mediapipe 0.10+ moved solutions under mediapipe.python.solutions
        from mediapipe.python.solutions import pose as mp_pose
    except ImportError:
        try:
            import mediapipe as mp
            mp_pose = mp.solutions.pose
        except (ImportError, AttributeError) as e:
            logger.warning("Pose extraction requires mediapipe: %s", e)
            return None

    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        logger.error("Could not open video: %s", video_path)
        return None

    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)) or 1
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)) or 1
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) or 0

    frames_out: list[FramePose] = []
    frame_idx = 0

    with mp_pose.Pose(
        static_image_mode=False,
        model_complexity=1,
        min_detection_confidence=0.5,
        min_tracking_confidence=0.5,
    ) as pose:
        while True:
            ret, image = cap.read()
            if not ret:
                break
            timestamp = frame_idx / fps if fps > 0 else float(frame_idx)
            rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            results = pose.process(rgb)

            keypoints: list[Keypoint] = []
            if results.pose_landmarks:
                lm = results.pose_landmarks.landmark
                # Add neck as midpoint of shoulders
                if len(lm) > 12:
                    lx = lm[11].x * w
                    ly = lm[11].y * h
                    rx = lm[12].x * w
                    ry = lm[12].y * h
                    keypoints.append(
                        Keypoint(
                            joint="neck",
                            x=(lx + rx) / 2.0,
                            y=(ly + ry) / 2.0,
                        )
                    )
                for idx, name in MEDIAPIPE_TO_JOINT:
                    if idx < len(lm):
                        keypoints.append(
                            Keypoint(
                                joint=name,
                                x=lm[idx].x * w,
                                y=lm[idx].y * h,
                            )
                        )
            frames_out.append(
                FramePose(frame=frame_idx, timestamp=timestamp, keypoints=keypoints)
            )
            frame_idx += 1
            if frame_count > 0 and frame_idx >= frame_count:
                break

    cap.release()
    out_fps = fps_out if fps_out and fps_out > 0 else fps
    return MotionSequence(fps=out_fps, width=w, height=h, frames=frames_out)


def extract_poses_from_video(video_path: Path, fps_out: float | None = 30.0) -> MotionSequence | None:
    """
    Extract body skeleton keypoints from every frame of a video.
    Returns MotionSequence (frame, timestamp, keypoints per frame) or None if failed.
    """
    path = Path(video_path)
    if not path.exists() or not path.is_file():
        logger.error("Video file not found: %s", path)
        return None
    return _extract_poses_mediapipe(path, fps_out=fps_out)


def normalize_motion(motion: MotionSequence) -> MotionSequence:
    """
    Normalize pose sequence for model input:
    - scale normalization (to unit box or fixed scale)
    - center alignment (centroid to origin)
    - rotation stabilization (optional: align principal axis)
    """
    if not motion.frames:
        return motion

    import math

    all_x: list[float] = []
    all_y: list[float] = []
    for fp in motion.frames:
        for kp in fp.keypoints:
            all_x.append(kp.x)
            all_y.append(kp.y)
    if not all_x:
        return motion

    min_x, max_x = min(all_x), max(all_x)
    min_y, max_y = min(all_y), max(all_y)
    span_x = max_x - min_x or 1.0
    span_y = max_y - min_y or 1.0
    scale = max(span_x, span_y) or 1.0
    cx = (min_x + max_x) / 2.0
    cy = (min_y + max_y) / 2.0

    normalized_frames: list[FramePose] = []
    for fp in motion.frames:
        new_kps = [
            Keypoint(joint=kp.joint, x=(kp.x - cx) / scale, y=(kp.y - cy) / scale)
            for kp in fp.keypoints
        ]
        normalized_frames.append(
            FramePose(frame=fp.frame, timestamp=fp.timestamp, keypoints=new_kps)
        )

    return MotionSequence(
        fps=motion.fps,
        width=motion.width,
        height=motion.height,
        frames=normalized_frames,
    )
