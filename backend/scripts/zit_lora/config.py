"""
Z-Image-Turbo LoRA 학습 설정.
- 지정 HF 데이터셋만 사용, merge/추가 수집 금지.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

# 스타일별 단일 HF 데이터셋 (merge 금지)
STYLE_DATASET: dict[str, str] = {
    "3d_render": "Guizmus/3DChanStyle",
    "cyberpunk": "imadjinn/cyberpunk_diffusion",
    "pixel_art": "nightaway/PixelArt2",
}

# caption 없을 때 사용할 스타일 토큰만
STYLE_TOKEN: dict[str, str] = {
    "3d_render": "3d_render style",
    "cyberpunk": "cyberpunk style",
    "pixel_art": "pixel art style",
}

@dataclass(frozen=True)
class StyleConfig:
    style_key: str
    rank: int
    alpha: int
    epochs: int

STYLE_CONFIGS: dict[str, StyleConfig] = {
    "3d_render": StyleConfig("3d_render", rank=16, alpha=16, epochs=6),
    "cyberpunk": StyleConfig("cyberpunk", rank=16, alpha=16, epochs=6),
    "pixel_art": StyleConfig("pixel_art", rank=8, alpha=8, epochs=8),
}

RESOLUTION = 1024
ZIT_MODEL_ID = "Tongyi-MAI/Z-Image-Turbo"
LEARNING_RATE = 1e-4
OPTIMIZER = "adamw_8bit"
WEIGHT_DECAY = 0.01

# 출력: lora_output/{style}.safetensors
BACKEND_DIR = Path(__file__).resolve().parent.parent.parent
LORA_OUTPUT_DIR = BACKEND_DIR / "lora_output"
