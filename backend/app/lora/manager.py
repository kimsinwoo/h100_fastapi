"""
Dynamic LoRA loading and application. Cache by path, scale control.
PEFT 전체 저장(old) 형식 → diffusers(transformer.* LoRA만) 로드 시 변환 지원.
"""

from __future__ import annotations

import logging
import threading
from pathlib import Path
from typing import Any

from app.core.config import get_settings

logger = logging.getLogger(__name__)

_lock = threading.RLock()
_loaded: dict[str, list[str]] = {}

PEFT_PREFIX = "base_model.model."


def _load_safetensors_state_dict(path: Path) -> dict[str, Any]:
    import safetensors.torch
    with safetensors.torch.safe_open(path, framework="pt", device="cpu") as f:
        return {k: f.get_tensor(k) for k in f.keys()}


def _is_old_peft_format(state_dict: dict[str, Any]) -> bool:
    if not state_dict:
        return False
    keys = list(state_dict.keys())
    return any("base_model.model." in k for k in keys)


def _convert_old_peft_to_diffusers(state_dict: dict[str, Any]) -> dict[str, Any]:
    """PEFT 전체 state_dict에서 LoRA 키만 남기고, transformer 내부 경로만 사용 (접두사 없음)."""
    out = {}
    for k, v in state_dict.items():
        if not k.startswith(PEFT_PREFIX):
            continue
        if "lora_A" not in k and "lora_B" not in k:
            continue
        new_key = k.replace(PEFT_PREFIX, "", 1)
        out[new_key] = v
    return out


def _ensure_transformer_internal_keys(state_dict: dict[str, Any]) -> dict[str, Any]:
    """LoRA 키에서 'transformer.' 접두사가 있으면 제거해 diffusers가 transformer 내부에서 매칭하도록 함."""
    prefix = "transformer."
    out = {}
    for k, v in state_dict.items():
        if "lora_A" not in k and "lora_B" not in k:
            continue
        new_key = k[len(prefix):] if k.startswith(prefix) else k
        out[new_key] = v
    return out if out else state_dict


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

    raw_dict = _load_safetensors_state_dict(path)
    if _is_old_peft_format(raw_dict):
        state_dict = _convert_old_peft_to_diffusers(raw_dict)
        if not state_dict:
            sample = list(raw_dict.keys())[:15]
            logger.warning(
                "LoRA file has base_model.model.* but no lora_A/lora_B keys. "
                "Sample keys: %s. Re-train with updated train_lora_zit.py.",
                sample,
            )
            raise ValueError(
                "LoRA file is old format without lora_A/lora_B. "
                "Re-train with current train_lora_zit.py to save diffusers-compatible LoRA."
            )
        logger.info("Converted old PEFT LoRA to diffusers format (%d keys)", len(state_dict))
        pipe.load_lora_weights(
            state_dict,
            adapter_name=name,
            lora_scale=scale,
        )
    else:
        # 이미 diffusers 형식이어도 키에 "transformer." 접두사가 있으면 제거 (Target modules not found 방지)
        state_dict = _ensure_transformer_internal_keys(raw_dict)
        if state_dict:
            pipe.load_lora_weights(
                state_dict,
                adapter_name=name,
                lora_scale=scale,
            )
        else:
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
