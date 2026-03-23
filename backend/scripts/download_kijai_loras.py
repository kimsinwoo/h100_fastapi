#!/usr/bin/env python3
"""
Kijai Wan LoRA → 레포 루트 ComfyUI/models/loras/
zimage_webapp/backend 에서 실행해도 됩니다 (상위 talktailForPet/ComfyUI 를 찾음).

  cd zimage_webapp/backend
  python3 scripts/download_kijai_loras.py

직접 ComfyUI 에서 쓰려면: ComfyUI/scripts/download_kijai_loras.py
"""
from __future__ import annotations

import os
import sys
from pathlib import Path

# backend/scripts -> backend -> zimage_webapp -> talktailForPet
_REPO_ROOT = Path(__file__).resolve().parent.parent.parent.parent
_SCRIPT = _REPO_ROOT / "ComfyUI" / "scripts" / "download_kijai_loras.py"


def main() -> None:
    if not _SCRIPT.is_file():
        print(
            f"ERROR: {_SCRIPT} 없습니다.\n"
            "  talktailForPet/ComfyUI/scripts/download_kijai_loras.py 가 있어야 합니다.",
            file=sys.stderr,
        )
        raise SystemExit(1)
    os.execv(sys.executable, [sys.executable, str(_SCRIPT)])


if __name__ == "__main__":
    main()
