#!/usr/bin/env python3
"""
LoRA 학습 진입점.
환경변수: TRAINING_DATA_DIR, TRAINING_IMAGES_DIR, TRAINING_DATASET_JSON
- dataset.json 과 images/ 를 읽어서 Kohya_ss / diffusers 형식으로 준비한 뒤
  실제 학습은 아래 _run_training() 에서 수행합니다.
- LoRA 학습 라이브러리(kohya_ss, diffusers train script 등)가 설치되어 있으면
  이 스크립트를 수정해 연동하면 됩니다.
"""
from __future__ import annotations

import json
import os
import shutil
import sys
from pathlib import Path


def main() -> int:
    data_dir = os.environ.get("TRAINING_DATA_DIR", "")
    images_dir = os.environ.get("TRAINING_IMAGES_DIR", "")
    dataset_json = os.environ.get("TRAINING_DATASET_JSON", "")

    if not dataset_json or not Path(dataset_json).exists():
        print("TRAINING_DATASET_JSON not set or file missing", file=sys.stderr)
        return 1

    with open(dataset_json, "r", encoding="utf-8") as f:
        items = json.load(f)

    if not items:
        print("No items in dataset", file=sys.stderr)
        return 1

    # Kohya_ss / Every Dream 2 형식: 각 이미지와 같은 이름의 .txt 에 캡션
    prepared_dir = Path(data_dir) / "prepared_for_training"
    prepared_dir.mkdir(parents=True, exist_ok=True)
    images_path = Path(images_dir)

    for i, it in enumerate(items):
        fn = it.get("image_filename", "")
        caption = (it.get("caption") or "").strip() or "image"
        src = images_path / fn
        if not src.exists():
            continue
        # 파일명 충돌 방지: index_originalname
        stem = f"{i}_{Path(fn).stem}"
        dest_img = prepared_dir / f"{stem}.png"
        dest_txt = prepared_dir / f"{stem}.txt"
        shutil.copy2(src, dest_img)
        dest_txt.write_text(caption, encoding="utf-8")

    print(f"Prepared {len(items)} images + captions at: {prepared_dir}")

    # 실제 LoRA 학습 호출 (선택)
    # - kohya_ss: subprocess.run(["python", "train_network.py", ...])
    # - diffusers: from diffusers import ... ; trainer.train()
    success = _run_training(str(prepared_dir), data_dir)
    return 0 if success else 1


def _run_training(prepared_dir: str, data_dir: str) -> bool:
    """
    LoRA 학습 실행.
    현재는 플레이스홀더: prepared 디렉터리만 만들어 두고,
    여기에 Kohya_ss / diffusers 학습 코드를 연동하면 됩니다.
    """
    print("LoRA training step: placeholder. To run real training, edit scripts/run_lora_training.py and call your trainer (e.g. Kohya_ss train_network.py).")
    print("Prepared dataset is at:", prepared_dir)
    return True


if __name__ == "__main__":
    sys.exit(main())
