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


def _get_mediapipe_pose_module():
    """Return the MediaPipe pose module (with .Pose class) for the current install. None if unavailable."""
    _last_import_error: Exception | None = None

    # 1) mediapipe.python.solutions.pose (레거시, 패키지 내부 경로)
    try:
        from mediapipe.python.solutions import pose as mp_pose
        return mp_pose
    except Exception as e1:
        _last_import_error = e1

    # 2) 부모 패키지 먼저 로드 후 pose 로드 (일부 환경에서 lazy load 이슈 방지)
    try:
        import mediapipe.python.solutions  # noqa: F401
        from mediapipe.python.solutions import pose as mp_pose
        return mp_pose
    except Exception:
        pass

    # 3) importlib.import_module로 로드 (sys.modules 등록으로 하위 import 해결)
    try:
        import importlib
        mod = importlib.import_module("mediapipe.python.solutions.pose")
        if mod is not None and hasattr(mod, "Pose"):
            return mod
    except Exception:
        pass

    # 4) mediapipe.solutions.pose (일부 배포판은 python 없이 solutions만 노출)
    try:
        from mediapipe.solutions import pose as mp_pose
        return mp_pose
    except ImportError:
        pass

    # 5) import mediapipe 후 속성 체인으로 접근 (mp.python.solutions.pose)
    try:
        import mediapipe as mp
        chain = getattr(mp, "python", None)
        if chain is not None:
            chain = getattr(chain, "solutions", None)
        if chain is None:
            chain = getattr(mp, "solutions", None)
        if chain is not None:
            pose_mod = getattr(chain, "pose", None)
            if pose_mod is not None and hasattr(pose_mod, "Pose"):
                return pose_mod
    except (ImportError, AttributeError):
        pass

    # 6) 실패 시: mediapipe.tasks(PoseLandmarker) 폴백 사용 안내
    try:
        import mediapipe as mp
        if "tasks" in dir(mp):
            logger.info(
                "mediapipe 레거시 Pose 없음(dir=%s). mediapipe.tasks(PoseLandmarker)로 포즈 추출 시도.",
                [x for x in dir(mp) if not x.startswith("_")],
            )
        else:
            logger.warning(
                "mediapipe는 설치되어 있으나 Pose 모듈을 찾지 못함. mediapipe.__file__=%s, dir(mediapipe)=%s. 오류: %s",
                getattr(mp, "__file__", None),
                [x for x in dir(mp) if not x.startswith("_")],
                _last_import_error or "unknown",
            )
    except ImportError:
        logger.warning(
            "mediapipe 패키지 자체를 import할 수 없음. pip install mediapipe>=0.10.0 확인. 오류: %s",
            _last_import_error or "unknown",
        )
    return None


def _extract_poses_mediapipe(video_path: Path, fps_out: float | None = None) -> MotionSequence | None:
    """Extract body keypoints per frame using MediaPipe Pose. Returns None if dependency missing."""
    try:
        import cv2
    except ImportError as e:
        logger.warning("Pose extraction requires opencv-python: %s", e)
        return None

    mp_pose = _get_mediapipe_pose_module()
    if mp_pose is None:
        logger.info(
            "Pose (legacy) not available; extract_poses_from_video will try MediaPipe Tasks API next."
        )
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


# MediaPipe Tasks API (mediapipe.tasks) — 레거시 mediapipe.python 없을 때 사용
POSE_LANDMARKER_MODEL_URL = (
    "https://storage.googleapis.com/mediapipe-models/pose_landmarker/pose_landmarker_heavy/float16/1/pose_landmarker_heavy.task"
)


def _get_pose_landmarker_model_path() -> Path | None:
    """Tasks API용 .task 모델 경로. 없으면 다운로드 시도 후 반환. 실패 시 None."""
    from app.core.config import get_settings
    settings = get_settings()
    if getattr(settings, "pose_landmarker_model_path", None):
        p = Path(settings.pose_landmarker_model_path)
        if p.exists():
            return p
        logger.warning("POSE_LANDMARKER_MODEL_PATH not found: %s", p)
    default = settings.pose_cache_dir / "pose_landmarker.task"
    if default.exists():
        return default
    try:
        import urllib.request
        default.parent.mkdir(parents=True, exist_ok=True)
        logger.info("Downloading PoseLandmarker model to %s ...", default)
        urllib.request.urlretrieve(POSE_LANDMARKER_MODEL_URL, default)
        return default
    except Exception as e:
        logger.warning("PoseLandmarker model download failed: %s. Set POSE_LANDMARKER_MODEL_PATH to a .task file.", e)
        return None


def _extract_poses_mediapipe_tasks(video_path: Path, fps_out: float | None = None) -> MotionSequence | None:
    """mediapipe.tasks.vision.PoseLandmarker (Tasks API)로 프레임별 포즈 추출. 레거시 solutions.pose 없을 때 사용."""
    try:
        import cv2
    except ImportError as e:
        logger.warning("Pose extraction (Tasks) requires opencv-python: %s", e)
        return None
    model_path = _get_pose_landmarker_model_path()
    if not model_path or not model_path.exists():
        logger.debug("PoseLandmarker model not found; _get_pose_landmarker_model_path returned %s", model_path)
        return None
    try:
        from mediapipe.tasks import python as mp_tasks_python
        from mediapipe.tasks.python import vision
        BaseOptions = mp_tasks_python.BaseOptions
        RunningMode = vision.RunningMode
    except ImportError as e:
        logger.warning("MediaPipe Tasks API not available: %s", e)
        return None
    try:
        import mediapipe as mp
    except ImportError:
        return None
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        logger.error("Could not open video: %s", video_path)
        return None
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)) or 1
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)) or 1
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) or 0
    base_options = BaseOptions(model_asset_path=str(model_path))
    options = vision.PoseLandmarkerOptions(
        base_options=base_options,
        running_mode=RunningMode.VIDEO,
        min_pose_detection_confidence=0.5,
        min_tracking_confidence=0.5,
    )
    landmarker = vision.PoseLandmarker.create_from_options(options)
    frames_out: list[FramePose] = []
    frame_idx = 0
    try:
        while True:
            ret, image = cap.read()
            if not ret:
                break
            timestamp = frame_idx / fps if fps > 0 else float(frame_idx)
            timestamp_ms = int(timestamp * 1000)
            rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
            result = landmarker.detect_for_video(mp_image, timestamp_ms)
            keypoints: list[Keypoint] = []
            if result.pose_landmarks:
                for pose_landmarks in result.pose_landmarks:
                    lm = pose_landmarks
                    try:
                        n = len(lm)
                    except TypeError:
                        break
                    if n > 12:
                        p11 = lm[11]
                        p12 = lm[12]
                        lx = getattr(p11, "x", 0) * w
                        ly = getattr(p11, "y", 0) * h
                        rx = getattr(p12, "x", 0) * w
                        ry = getattr(p12, "y", 0) * h
                        keypoints.append(
                            Keypoint(joint="neck", x=(lx + rx) / 2.0, y=(ly + ry) / 2.0)
                        )
                    for idx, name in MEDIAPIPE_TO_JOINT:
                        if idx < n:
                            p = lm[idx]
                            keypoints.append(
                                Keypoint(
                                    joint=name,
                                    x=getattr(p, "x", 0) * w,
                                    y=getattr(p, "y", 0) * h,
                                )
                            )
                    break
            frames_out.append(FramePose(frame=frame_idx, timestamp=timestamp, keypoints=keypoints))
            frame_idx += 1
            if frame_count > 0 and frame_idx >= frame_count:
                break
    finally:
        landmarker.close()
    cap.release()
    out_fps = fps_out if fps_out and fps_out > 0 else fps
    if not frames_out:
        return None
    logger.info("Pose extraction (Tasks API) OK: %d frames from %s", len(frames_out), video_path.name)
    return MotionSequence(fps=out_fps, width=w, height=h, frames=frames_out)


def extract_poses_from_video(video_path: Path, fps_out: float | None = 30.0) -> MotionSequence | None:
    """
    Extract body skeleton keypoints from every frame of a video.
    Returns MotionSequence (frame, timestamp, keypoints per frame) or None if failed.
    레거시 mediapipe.python.solutions.pose → 없으면 mediapipe.tasks.vision.PoseLandmarker 사용.
    """
    path = Path(video_path)
    if not path.exists() or not path.is_file():
        logger.error("Video file not found: %s", path)
        return None
    out = _extract_poses_mediapipe(path, fps_out=fps_out)
    if out is not None:
        return out
    logger.info("Trying pose extraction with MediaPipe Tasks API (PoseLandmarker)...")
    out = _extract_poses_mediapipe_tasks(path, fps_out=fps_out)
    if out is None:
        logger.warning(
            "Pose extraction (Tasks API) failed: need .task model (auto-download or POSE_LANDMARKER_MODEL_PATH). "
            "See data/pose_cache/pose_landmarker.task or env POSE_LANDMARKER_MODEL_PATH."
        )
    return out


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
