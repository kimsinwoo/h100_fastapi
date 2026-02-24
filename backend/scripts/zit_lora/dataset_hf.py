"""
HF 데이터셋 직접 로드. 로컬 export 금지. resize + center crop만.
augmentation( color jitter, random flip 등) 금지.
"""
from __future__ import annotations

import logging
from typing import Any

from PIL import Image
import torch
from torch.utils.data import Dataset

from config import STYLE_DATASET, STYLE_TOKEN, RESOLUTION

logger = logging.getLogger(__name__)


def _detect_image_column(columns: list[str], row: dict[str, Any]) -> str | None:
    for col in columns:
        if col.lower() in ("caption", "text", "prompt"):
            continue
        v = row.get(col)
        if v is None:
            continue
        if hasattr(v, "size") and hasattr(v, "mode"):
            return col
        if isinstance(v, dict) and "path" in v:
            return col
        if isinstance(v, list) and len(v) > 0:
            return col
    return None


def _detect_caption_column(columns: list[str], row: dict[str, Any]) -> str | None:
    for col in ("caption", "text", "prompt", "captions", "title"):
        if col in columns:
            v = row.get(col)
            if isinstance(v, str) and v.strip():
                return col
            if isinstance(v, list) and len(v) > 0 and isinstance(v[0], str):
                return col
    return None


def _pil_from_value(v: Any) -> Image.Image | None:
    if v is None:
        return None
    if hasattr(v, "size") and hasattr(v, "mode"):
        return v.convert("RGB") if v.mode != "RGB" else v
    if isinstance(v, dict) and "path" in v:
        return Image.open(v["path"]).convert("RGB")
    if isinstance(v, list) and len(v) > 0 and hasattr(v[0], "__iter__"):
        import numpy as np
        arr = np.array(v)
        if arr.ndim >= 2:
            return Image.fromarray(arr).convert("RGB")
    return None


def _caption_from_value(v: Any, style_token: str) -> str:
    if isinstance(v, str) and v.strip():
        return v.strip()
    if isinstance(v, list) and len(v) > 0 and isinstance(v[0], str):
        return v[0].strip()
    return style_token


def _resize_center_crop(img: Image.Image, size: int) -> Image.Image:
    w, h = img.size
    if w == h == size:
        return img
    scale = size / min(w, h)
    nw, nh = round(w * scale), round(h * scale)
    img = img.resize((nw, nh), Image.Resampling.LANCZOS)
    left = (nw - size) // 2
    top = (nh - size) // 2
    return img.crop((left, top, left + size, top + size))


class HFDatasetWrapper(Dataset[dict[str, Any]]):
    """
    datasets.load_dataset(..., streaming=False)으로 직접 로드.
    로컬 export 없음. image 컬럼 자동 감지, caption 없으면 스타일 토큰만 사용.
    resize + center crop만 적용. augmentation 금지.
    """

    def __init__(self, style_key: str, resolution: int = RESOLUTION) -> None:
        if style_key not in STYLE_DATASET:
            raise ValueError(f"Unknown style_key: {style_key}. Allowed: {list(STYLE_DATASET.keys())}")
        self.style_key = style_key
        self.resolution = resolution
        self.style_token = STYLE_TOKEN[style_key]
        repo = STYLE_DATASET[style_key]
        from datasets import load_dataset
        self._ds = load_dataset(repo, split="train", streaming=False)
        cols = self._ds.column_names
        image_col = None
        caption_col = None
        for i in range(min(5, len(self._ds))):
            row = self._ds[i]
            if image_col is None:
                image_col = _detect_image_column(cols, row)
            if caption_col is None:
                caption_col = _detect_caption_column(cols, row)
            if image_col is not None:
                break
        if image_col is None:
            raise RuntimeError(f"No image column found in {repo}. Columns: {cols}")
        self._image_col = image_col
        self._caption_col = caption_col
        self._length = len(self._ds)
        logger.info("HF dataset %s: image_col=%s caption_col=%s len=%d", repo, image_col, caption_col, self._length)

    def __len__(self) -> int:
        return self._length

    def __getitem__(self, idx: int) -> dict[str, Any]:
        row = self._ds[idx]
        pil = _pil_from_value(row.get(self._image_col))
        if pil is None:
            raise ValueError(f"Row {idx}: could not get PIL image from column {self._image_col}")
        pil = _resize_center_crop(pil, self.resolution)
        cap = _caption_from_value(row.get(self._caption_col) if self._caption_col else None, self.style_token)
        return {"image": pil, "caption": cap}
