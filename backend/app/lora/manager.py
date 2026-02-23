"""
Dynamic LoRA loading and application. Cache by path, scale control.
"""

from __future__ import annotations

import logging
import threading
from pathlib import Path

from app.core.config import get_settings

logger = logging.getLogger(__name__)

_lock = threading.RLock()
_loaded: dict[str, list[str]] = {}


def load_lora(
    pipe: object,
    lora_path: str | Path,
    scale: float = 0.85,
    adapter_name: str | None = None,
) -> None:
    path = Path(lora_path)
    if not path.suffix:
        path = path.with_suffix(".safetensors")
    if not path.exists():
        raise FileNotFoundError(f"LoRA not found: {path}")
    key = str(path.resolve())
    name = adapter_name or f"lora_{path.stem}"

    if not hasattr(pipe, "load_lora_weights"):
        logger.warning("Pipeline does not support load_lora_weights")
        return

    with _lock:
        pipe_id = id(pipe)
        pid = str(pipe_id)
        if pid not in _loaded:
            _loaded[pid] = []
        if name in _loaded[pid]:
            return

    weight_name = path.name if path.suffix == ".safetensors" else None
    pipe.load_lora_weights(
        str(path.parent),
        weight_name=weight_name or path.name,
        adapter_name=name,
        lora_scale=scale,
    )
    with _lock:
        _loaded.setdefault(pid, []).append(name)


def unload_lora(pipe: object, adapter_name: str | None = None) -> None:
    with _lock:
        pid = str(id(pipe))
        if pid not in _loaded or not _loaded[pid]:
            return
        name = adapter_name or (_loaded[pid][-1] if _loaded[pid] else None)
    if name and hasattr(pipe, "unload_lora_weights"):
        try:
            pipe.unload_lora_weights(name)
        except Exception as e:
            logger.warning("Unload LoRA %s: %s", name, e)
        with _lock:
            if pid in _loaded and name in _loaded[pid]:
                _loaded[pid].remove(name)


def list_available_loras() -> list[str]:
    d = get_settings().lora_dir
    out: list[str] = []
    for f in d.glob("*.safetensors"):
        out.append(f.stem)
    return sorted(out)
