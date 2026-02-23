"""
SDXL LoRA training: prepare dataset and run training script. Saves safetensors + metadata.
"""

from __future__ import annotations

import asyncio
import json
import logging
import subprocess
import sys
from pathlib import Path

from app.core.config import get_settings
from app.training.dataset import load_captions

logger = logging.getLogger(__name__)


def train_lora(
    dataset_path: str | Path,
    output_path: str | Path,
    rank: int = 4,
    learning_rate: float = 1e-4,
    steps: int = 500,
    batch_size: int = 1,
    resolution: int = 1024,
    resume_from: str | Path | None = None,
) -> dict[str, str | int | float]:
    """
    Prepare dataset and run LoRA training. Saves safetensors and metadata.json.
    Uses diffusers training script via subprocess. Returns summary dict.
    """
    dataset_path = Path(dataset_path)
    output_path = Path(output_path)
    output_path.mkdir(parents=True, exist_ok=True)

    pairs = load_captions(dataset_path)
    if not pairs:
        raise ValueError("No image-caption pairs in dataset")

    s = get_settings()
    backend_dir = Path(__file__).resolve().parent.parent.parent
    script_dir = backend_dir / "scripts"
    train_script = script_dir / "train_lora_sdxl.py"
    if not train_script.exists():
        (output_path / "metadata.json").write_text(
            json.dumps({
                "error": "train_lora_sdxl.py not found",
                "dataset_path": str(dataset_path),
                "output_path": str(output_path),
                "steps": steps,
            }, indent=2),
            encoding="utf-8",
        )
        return {"output_path": str(output_path), "steps": 0, "status": "error", "error": "train_lora_sdxl.py not found"}

    cmd = [
        sys.executable,
        str(train_script),
        "--dataset_path", str(dataset_path),
        "--output_path", str(output_path),
        "--rank", str(rank),
        "--learning_rate", str(learning_rate),
        "--max_train_steps", str(steps),
        "--train_batch_size", str(batch_size),
        "--resolution", str(resolution),
    ]
    if resume_from:
        cmd += ["--resume_from", str(resume_from)]

    env = __import__("os").environ.copy()
    env["HF_HOME"] = str(Path.home() / ".cache" / "huggingface")

    try:
        proc = subprocess.run(cmd, cwd=str(script_dir), env=env, capture_output=True, text=True, timeout=86400)
        if proc.returncode != 0:
            logger.error("Training failed: %s", proc.stderr[-2000:] if proc.stderr else proc.stdout)
            return {"output_path": str(output_path), "steps": 0, "status": "failed", "stderr": (proc.stderr or "")[-500:]}
    except subprocess.TimeoutExpired:
        return {"output_path": str(output_path), "steps": 0, "status": "timeout"}
    except Exception as e:
        return {"output_path": str(output_path), "steps": 0, "status": "error", "error": str(e)}

    meta = {
        "rank": rank,
        "learning_rate": learning_rate,
        "steps": steps,
        "base_model": s.sdxl_base_id,
    }
    (output_path / "metadata.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")
    return {"output_path": str(output_path), "steps": steps, "status": "completed"}


async def train_lora_async(
    dataset_path: str | Path,
    output_path: str | Path,
    rank: int = 4,
    learning_rate: float = 1e-4,
    steps: int = 500,
    batch_size: int = 1,
    resolution: int = 1024,
    resume_from: str | Path | None = None,
) -> dict[str, str | int | float]:
    loop = asyncio.get_event_loop()
    return await loop.run_in_executor(
        None,
        lambda: train_lora(dataset_path, output_path, rank, learning_rate, steps, batch_size, resolution, resume_from),
    )
