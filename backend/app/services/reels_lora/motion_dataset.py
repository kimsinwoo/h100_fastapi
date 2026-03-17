"""
Motion Dataset for AnimateDiff LoRA training.

Converts VideoClip objects into PyTorch tensors suitable for training:
  - pixel_values : (T, C, H, W) float32 in [-1, 1]
  - pose_frames  : (T, C, H, W) float32 in [-1, 1]  (pose skeleton images)
  - motion_ids   : list[int]                          (frame-level keypoints for optional conditioning)

Temporal augmentation:
  - Random temporal offset (jitter) within each clip
  - Random horizontal flip (pose keypoints mirrored accordingly)
  - Temporal speed perturbation (sub-sample at different strides)
"""

from __future__ import annotations

import logging
import random
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import cv2
import numpy as np
import torch
from torch.utils.data import Dataset

from .dataset_pipeline import FrameData, VideoClip, render_skeleton, PoseKeypoints

logger = logging.getLogger(__name__)

# ──────────────────────────────────────────────────────────────────────────────
# Normalization helpers
# ──────────────────────────────────────────────────────────────────────────────

def bgr_to_tensor(img: np.ndarray) -> torch.Tensor:
    """(H, W, 3) BGR uint8  →  (3, H, W) float32 in [-1, 1]."""
    rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB).astype(np.float32) / 127.5 - 1.0
    return torch.from_numpy(rgb.transpose(2, 0, 1))  # (3, H, W)


def normalize_keypoints_temporal(
    seq: List[np.ndarray],
) -> List[np.ndarray]:
    """
    Temporal normalization of (18, 3) body keypoint sequences.

    Strategy:
      1. Root-center: subtract hip midpoint so translation is removed
      2. Scale-normalize: divide by torso height so scale is removed
      3. Only normalize valid frames (confidence > 0.1)

    Args:
        seq: list of (18, 3) arrays  [x, y, conf] normalized 0-1

    Returns:
        Normalized list with same shape.
    """
    normalized: List[np.ndarray] = []

    for kps in seq:
        kps = kps.copy()
        valid = kps[:, 2] > 0.1

        if valid.sum() < 4:
            normalized.append(kps)
            continue

        # Root: midpoint of hips (COCO-18 indices 11, 12)
        left_hip = kps[11, :2] if kps[11, 2] > 0.1 else None
        right_hip = kps[12, :2] if kps[12, 2] > 0.1 else None
        if left_hip is not None and right_hip is not None:
            root = (left_hip + right_hip) / 2.0
        elif left_hip is not None:
            root = left_hip
        elif right_hip is not None:
            root = right_hip
        else:
            root = np.array([0.5, 0.5])

        kps[valid, :2] -= root

        # Scale: torso height (shoulder midpoint → hip midpoint)
        left_shoulder = kps[5, :2] if kps[5, 2] > 0.1 else None
        right_shoulder = kps[6, :2] if kps[6, 2] > 0.1 else None
        if left_shoulder is not None and right_shoulder is not None:
            shoulder_mid = (left_shoulder + right_shoulder) / 2.0
        else:
            shoulder_mid = None

        if shoulder_mid is not None:
            torso_h = float(np.linalg.norm(shoulder_mid - root))
            if torso_h > 1e-3:
                kps[valid, :2] /= torso_h

        normalized.append(kps)

    return normalized


def align_clip_length(frames: List[FrameData], target_len: int) -> List[FrameData]:
    """
    Ensure clip has exactly target_len frames.
    - If too long: sample evenly.
    - If too short: repeat last frame.
    """
    n = len(frames)
    if n == target_len:
        return frames
    if n > target_len:
        indices = np.linspace(0, n - 1, target_len, dtype=int)
        return [frames[i] for i in indices]
    # Pad with last frame
    padded = list(frames)
    while len(padded) < target_len:
        padded.append(frames[-1])
    return padded


# ──────────────────────────────────────────────────────────────────────────────
# Augmentation
# ──────────────────────────────────────────────────────────────────────────────

def temporal_jitter(frames: List[FrameData], max_offset: int = 2) -> List[FrameData]:
    """Random start offset for temporal augmentation."""
    if len(frames) <= max_offset:
        return frames
    offset = random.randint(0, max_offset)
    return frames[offset:]


def temporal_speed_perturb(frames: List[FrameData], target_len: int) -> List[FrameData]:
    """
    Randomly sub-sample at slightly different speeds (0.75x – 1.25x).
    Then align to target_len.
    """
    n = len(frames)
    speed = random.uniform(0.75, 1.25)
    new_len = min(n, max(target_len, int(target_len * speed)))
    indices = np.linspace(0, n - 1, new_len, dtype=int)
    resampled = [frames[i] for i in indices]
    return align_clip_length(resampled, target_len)


def horizontal_flip_frame(fd: FrameData) -> FrameData:
    """Flip image and mirror pose keypoints."""
    flipped_img = cv2.flip(fd.image.copy(), 1)  # 1 = horizontal

    flipped_pose = None
    flipped_pose_img = None
    if fd.pose is not None:
        kps = fd.pose.body.copy()
        # Mirror x: x' = 1 - x
        valid = kps[:, 2] > 0.05
        kps[valid, 0] = 1.0 - kps[valid, 0]

        # Swap left ↔ right keypoints (COCO-18 symmetric pairs)
        SWAP_PAIRS = [(1, 2), (3, 4), (5, 6), (7, 8), (9, 10), (11, 12), (13, 14), (15, 16)]
        for a, b in SWAP_PAIRS:
            kps[a], kps[b] = kps[b].copy(), kps[a].copy()

        flipped_pose = PoseKeypoints(body=kps)
        W, H = fd.image.shape[1], fd.image.shape[0]
        flipped_pose_img = render_skeleton(flipped_pose, W, H)

    return FrameData(
        frame_idx=fd.frame_idx,
        timestamp=fd.timestamp,
        image=flipped_img,
        pose=flipped_pose,
        pose_image=flipped_pose_img,
    )


# ──────────────────────────────────────────────────────────────────────────────
# Dataset
# ──────────────────────────────────────────────────────────────────────────────

class MotionDataset(Dataset):
    """
    PyTorch Dataset for AnimateDiff motion LoRA training.

    Each item returns a dict:
      - "pixel_values"  : (T, 3, H, W) float32 [-1, 1]
      - "pose_values"   : (T, 3, H, W) float32 [-1, 1]
      - "keypoints"     : (T, 18, 3)   float32  (normalized)
      - "category"      : str
      - "clip_path"     : str
    """

    def __init__(
        self,
        clips: List[VideoClip],
        clip_frames: int = 16,
        image_size: int = 512,
        augment: bool = True,
        flip_prob: float = 0.5,
        temporal_jitter_max: int = 2,
        speed_perturb: bool = True,
    ):
        self.clips = clips
        self.clip_frames = clip_frames
        self.image_size = image_size
        self.augment = augment
        self.flip_prob = flip_prob
        self.temporal_jitter_max = temporal_jitter_max
        self.speed_perturb = speed_perturb

        if not clips:
            raise ValueError("MotionDataset received empty clips list.")

        logger.info(
            "[MotionDataset] %d clips | frames=%d | size=%d | augment=%s",
            len(clips), clip_frames, image_size, augment,
        )

    def __len__(self) -> int:
        return len(self.clips)

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        clip = self.clips[idx]
        frames = list(clip.frames)

        # ── Temporal augmentation ──────────────────────────────────────────
        if self.augment:
            if self.temporal_jitter_max > 0:
                frames = temporal_jitter(frames, self.temporal_jitter_max)
            if self.speed_perturb:
                frames = temporal_speed_perturb(frames, self.clip_frames)

        frames = align_clip_length(frames, self.clip_frames)

        # ── Spatial flip ──────────────────────────────────────────────────
        do_flip = self.augment and random.random() < self.flip_prob
        if do_flip:
            frames = [horizontal_flip_frame(f) for f in frames]

        # ── Build tensors ─────────────────────────────────────────────────
        pixel_list: List[torch.Tensor] = []
        pose_list: List[torch.Tensor] = []
        kp_list: List[np.ndarray] = []

        H = W = self.image_size

        # Pre-collect keypoints for temporal normalization
        raw_kps = []
        for fd in frames:
            if fd.pose is not None:
                raw_kps.append(fd.pose.body.copy())
            else:
                raw_kps.append(np.zeros((18, 3), dtype=np.float32))

        norm_kps = normalize_keypoints_temporal(raw_kps)

        for i, fd in enumerate(frames):
            # Resize image to (H, W)
            img = cv2.resize(fd.image, (W, H), interpolation=cv2.INTER_LINEAR)
            pixel_list.append(bgr_to_tensor(img))

            # Pose image
            if fd.pose_image is not None:
                pose_img = cv2.resize(fd.pose_image, (W, H), interpolation=cv2.INTER_LINEAR)
            else:
                # Render from normalized keypoints onto black canvas
                kp = PoseKeypoints(body=norm_kps[i])
                pose_img = render_skeleton(kp, W, H)
            pose_list.append(bgr_to_tensor(pose_img))

            kp_list.append(norm_kps[i])

        pixel_values = torch.stack(pixel_list, dim=0)   # (T, 3, H, W)
        pose_values = torch.stack(pose_list, dim=0)     # (T, 3, H, W)
        keypoints = torch.from_numpy(np.stack(kp_list, axis=0))  # (T, 18, 3)

        return {
            "pixel_values": pixel_values,
            "pose_values": pose_values,
            "keypoints": keypoints,
            "category": clip.category,
            "clip_path": str(clip.video_path),
        }


# ──────────────────────────────────────────────────────────────────────────────
# Collate function for DataLoader
# ──────────────────────────────────────────────────────────────────────────────

def motion_collate_fn(batch: List[Dict[str, Any]]) -> Dict[str, Any]:
    """Custom collate: stack tensors, keep string fields as lists."""
    return {
        "pixel_values": torch.stack([b["pixel_values"] for b in batch]),  # (B, T, 3, H, W)
        "pose_values": torch.stack([b["pose_values"] for b in batch]),    # (B, T, 3, H, W)
        "keypoints": torch.stack([b["keypoints"] for b in batch]),         # (B, T, 18, 3)
        "category": [b["category"] for b in batch],
        "clip_path": [b["clip_path"] for b in batch],
    }


# ──────────────────────────────────────────────────────────────────────────────
# Factory
# ──────────────────────────────────────────────────────────────────────────────

def build_dataloader(
    clips: List[VideoClip],
    clip_frames: int = 16,
    image_size: int = 512,
    batch_size: int = 2,
    num_workers: int = 4,
    augment: bool = True,
    pin_memory: bool = True,
) -> torch.utils.data.DataLoader:
    """Build a DataLoader from a list of VideoClips."""
    from torch.utils.data import DataLoader

    dataset = MotionDataset(
        clips=clips,
        clip_frames=clip_frames,
        image_size=image_size,
        augment=augment,
    )
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=pin_memory,
        collate_fn=motion_collate_fn,
        persistent_workers=num_workers > 0,
    )
