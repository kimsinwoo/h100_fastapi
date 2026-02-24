"""
LoRAManager: hot swap, preload LoRA paths, apply with scaling.
No pipeline reconstruction. Isolation per request (unload after use).
"""
from __future__ import annotations

import logging
import threading
from pathlib import Path

from app.sdxl_prod.config import get_settings

logger = logging.getLogger(__name__)
_lock = threading.RLock()
_preloaded_paths: dict[str, Path] = {}
_adapter_name_prefix = "sdxl_prod_"


def _lora_dir() -> Path:
    return get_settings().lora_dir


def _resolve_lora_path(lora_key: str) -> Path | None:
    d = _lora_dir()
    if not d.exists():
        return None
    candidates = [
        d / lora_key / "pytorch_lora_weights.safetensors",
        d / f"{lora_key}.safetensors",
        d / lora_key,
    ]
    for p in candidates:
        if p.exists():
            if p.is_dir():
                sf = p / "pytorch_lora_weights.safetensors"
                return sf if sf.exists() else p
            return p
    return None


def preload_lora_paths(lora_keys: list[str]) -> None:
    """Preload LoRA paths (validate existence). Call at startup."""
    with _lock:
        for key in lora_keys:
            path = _resolve_lora_path(key)
            if path is not None:
                _preloaded_paths[key] = path
                logger.info("LoRA preloaded: %s -> %s", key, path)
            else:
                logger.debug("LoRA not found (optional): %s", key)


def get_lora_path(lora_key: str) -> Path | None:
    """Return preloaded path or resolve on demand."""
    with _lock:
        if lora_key in _preloaded_paths:
            return _preloaded_paths[lora_key]
    path = _resolve_lora_path(lora_key)
    if path is not None:
        with _lock:
            _preloaded_paths[lora_key] = path
    return path


def apply_lora(
    pipe: object,
    lora_key: str,
    scale: float | None = None,
) -> None:
    """Load and apply LoRA to pipeline with scale. Fails if pipeline has no load_lora_weights."""
    path = get_lora_path(lora_key)
    if path is None:
        raise FileNotFoundError(f"LoRA not found: {lora_key}")
    scale = scale if scale is not None else get_settings().lora_default_scale
    adapter_name = f"{_adapter_name_prefix}{lora_key}"
    if not hasattr(pipe, "load_lora_weights"):
        raise AttributeError("Pipeline does not support load_lora_weights")
    pipe.load_lora_weights(
        str(path.resolve()) if path.is_file() else str(path.resolve()),
        adapter_name=adapter_name,
        lora_scale=scale,
    )
    if hasattr(pipe, "set_adapters") and hasattr(pipe, "get_list_adapters"):
        try:
            adapters = pipe.get_list_adapters()
            if adapters:
                pipe.set_adapters(adapters, adapter_weights=[scale])
        except Exception:
            pass
    logger.debug("LoRA applied: %s scale=%s", lora_key, scale)


def unload_lora(pipe: object, lora_key: str) -> None:
    """Unload LoRA adapter from pipeline for request isolation."""
    adapter_name = f"{_adapter_name_prefix}{lora_key}"
    if hasattr(pipe, "unload_lora_weights"):
        try:
            pipe.unload_lora_weights(adapter_name)
        except Exception as e:
            logger.debug("Unload LoRA %s: %s", lora_key, e)


def unload_all_loras(pipe: object) -> None:
    """Unload all adapters that we may have added (by prefix)."""
    if not hasattr(pipe, "unload_lora_weights"):
        return
    for key in list(_preloaded_paths.keys()):
        unload_lora(pipe, key)
