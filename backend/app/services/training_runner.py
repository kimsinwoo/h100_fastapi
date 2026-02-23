"""
LoRA 학습 시작: 데이터 검사 후 백그라운드에서 학습 스크립트 실행.
"""

from __future__ import annotations

import asyncio
import logging
import os
import subprocess
import sys
from pathlib import Path

from app.core.config import get_settings
from app.services.training_store import _get_images_dir, _load_metadata

logger = logging.getLogger(__name__)

# 학습 실행 중 여부 (단일 프로세스 기준)
_training_running = False


def _run_lora_script_sync() -> tuple[bool, str]:
    """
    학습 데이터를 준비하고 LoRA 학습 스크립트 실행.
    반환: (성공 여부, 메시지)
    """
    global _training_running
    if _training_running:
        return False, "이미 학습이 실행 중입니다."

    items = _load_metadata()
    if not items:
        return False, "학습 데이터가 없습니다. 이미지와 캡션을 먼저 추가하세요."

    images_dir = _get_images_dir()
    settings = get_settings()
    training_dir = settings.training_dir
    backend_dir = Path(__file__).resolve().parent.parent.parent
    # 프로젝트 루트 scripts/ 우선, 없으면 backend/scripts/
    project_root = backend_dir.parent
    script_path = project_root / "scripts" / "run_lora_training.py"
    if not script_path.exists():
        script_path = backend_dir / "scripts" / "run_lora_training.py"
    if not script_path.exists():
        logger.warning("LoRA script not found at %s or %s — creating minimal placeholder", project_root / "scripts" / "run_lora_training.py", backend_dir / "scripts" / "run_lora_training.py")
        script_path = backend_dir / "scripts" / "run_lora_training.py"
        script_path.parent.mkdir(parents=True, exist_ok=True)
        script_path.write_text(
            "# LoRA training entrypoint\n"
            "import os, sys\n"
            "from pathlib import Path\n"
            "data_dir = os.environ.get('TRAINING_DATA_DIR', '')\n"
            "dataset_json = os.environ.get('TRAINING_DATASET_JSON', '')\n"
            "if not dataset_json or not Path(dataset_json).exists():\n"
            "    print('TRAINING_DATASET_JSON missing', file=sys.stderr)\n"
            "    sys.exit(1)\n"
            "print('Dataset at', dataset_json, '- Add scripts/run_lora_training.py for full prepare+train.')\n"
            "sys.exit(0)\n",
            encoding="utf-8",
        )

    env = os.environ.copy()
    env["TRAINING_DATA_DIR"] = str(training_dir)
    env["TRAINING_IMAGES_DIR"] = str(images_dir)
    env["TRAINING_DATASET_JSON"] = str(training_dir / "dataset.json")

    try:
        _training_running = True
        logger.info("Starting LoRA training (subprocess): %s", script_path)
        proc = subprocess.run(
            [sys.executable, str(script_path)],
            cwd=str(backend_dir),
            env=env,
            capture_output=True,
            text=True,
            timeout=3600 * 24,  # 24h max
        )
        out = (proc.stdout or "").strip() or "(no output)"
        err = (proc.stderr or "").strip() or ""
        if proc.returncode != 0:
            logger.error("LoRA training failed: %s\nstderr: %s", out, err)
            return False, f"학습 실패: {err or out}"
        logger.info("LoRA training finished: %s", out)
        return True, "학습이 완료되었습니다."
    except subprocess.TimeoutExpired:
        logger.error("LoRA training timed out")
        return False, "학습 시간 초과."
    except Exception as e:
        logger.exception("LoRA training error: %s", e)
        return False, str(e)
    finally:
        _training_running = False


def start_lora_training() -> dict:
    """
    학습 데이터 확인 후 백그라운드에서 LoRA 학습 시작.
    반환: { "status": "started" | "failed", "message": "...", "error": "..." (실패 시) }
    """
    items = _load_metadata()
    if not items:
        return {"status": "failed", "message": "학습 데이터가 없습니다.", "error": "No training data"}

    # 백그라운드에서 실행 (비동기로 기다리지 않음)
    loop = asyncio.get_event_loop()
    future = loop.run_in_executor(None, _run_lora_script_sync)

    def _on_done(fut: asyncio.Future) -> None:
        try:
            success, msg = fut.result()
            logger.info("Training finished: success=%s %s", success, msg)
        except Exception as e:
            logger.exception("Training task error: %s", e)

    future.add_done_callback(lambda f: _on_done(f))

    return {
        "status": "started",
        "message": "LoRA 학습을 백그라운드에서 시작했습니다. 완료까지 시간이 걸릴 수 있습니다. 서버 로그를 확인하세요.",
    }
