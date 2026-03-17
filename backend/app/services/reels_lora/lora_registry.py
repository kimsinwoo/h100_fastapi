"""
LoRA Registry — manages all trained Reels dance LoRA models.

Directory layout:
    data/reels_lora/
    ├── tiktok_shuffle/
    │   ├── lora_weights.safetensors
    │   └── metadata.json
    ├── kpop_challenge/
    │   ├── lora_weights.safetensors
    │   └── metadata.json
    ├── hiphop_reels/
    │   └── ...
    └── viral_trend/
        └── ...

Features:
  - Scan registry root for available categories
  - Load / unload LoRA weights from safetensors
  - Hot-swap LoRA during inference (thread-safe)
  - Return model metadata
"""

from __future__ import annotations

import json
import logging
import threading
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional

import torch

logger = logging.getLogger(__name__)

# ──────────────────────────────────────────────────────────────────────────────
# Built-in dance categories
# ──────────────────────────────────────────────────────────────────────────────

BUILTIN_CATEGORIES = [
    "tiktok_shuffle",
    "kpop_challenge",
    "hiphop_reels",
    "viral_trend",
]

CATEGORY_DISPLAY_NAMES: Dict[str, str] = {
    "tiktok_shuffle":  "TikTok Shuffle",
    "kpop_challenge":  "K-Pop Challenge",
    "hiphop_reels":    "Hip-Hop Reels",
    "viral_trend":     "Viral Trend",
}

CATEGORY_PROMPTS: Dict[str, str] = {
    "tiktok_shuffle": (
        "anthropomorphic {character} performing tiktok shuffle dance, "
        "full body visible, energetic footwork, arms moving rhythmically, "
        "dynamic motion, high quality, cinematic"
    ),
    "kpop_challenge": (
        "anthropomorphic {character} performing k-pop synchronized dance, "
        "full body visible, precise choreography, expressive arms, "
        "dynamic motion, vibrant, high quality"
    ),
    "hiphop_reels": (
        "anthropomorphic {character} performing hip-hop dance moves, "
        "full body visible, street style, fluid body waves, "
        "dynamic motion, urban aesthetic, high quality"
    ),
    "viral_trend": (
        "anthropomorphic {character} performing viral reels dance, "
        "full body visible, trending moves, entertaining, "
        "dynamic motion, engaging, high quality"
    ),
}

NEGATIVE_PROMPT = (
    "blurry, low quality, watermark, text, deformed body, extra limbs, "
    "static, motionless, bad anatomy, ugly, worst quality"
)


# ──────────────────────────────────────────────────────────────────────────────
# Data models
# ──────────────────────────────────────────────────────────────────────────────

@dataclass
class LoRAEntry:
    """Metadata and state for one registered LoRA model."""
    category: str
    display_name: str
    weights_path: Path
    metadata: Dict[str, Any] = field(default_factory=dict)
    is_loaded: bool = False
    state_dict: Optional[Dict[str, torch.Tensor]] = field(default=None, repr=False)

    @property
    def lora_rank(self) -> int:
        return self.metadata.get("lora_rank", 32)

    @property
    def lora_alpha(self) -> int:
        return self.metadata.get("lora_alpha", 32)

    @property
    def base_model(self) -> str:
        return self.metadata.get("base_model", "runwayml/stable-diffusion-v1-5")

    @property
    def motion_adapter(self) -> str:
        return self.metadata.get("motion_adapter", "guoyww/animatediff-motion-adapter-v1-5-2")


@dataclass
class RegistryInfo:
    """Serializable summary of the registry state."""
    total: int
    categories: List[str]
    loaded: List[str]
    entries: List[Dict[str, Any]]


# ──────────────────────────────────────────────────────────────────────────────
# Registry
# ──────────────────────────────────────────────────────────────────────────────

class LoRARegistry:
    """
    Thread-safe registry for Reels dance LoRA models.

    Usage:
        registry = LoRARegistry(root=Path("data/reels_lora"))
        registry.scan()                           # discover available LoRAs
        entry = registry.get("tiktok_shuffle")    # get entry (lazy-load)
        sd = registry.load_weights("tiktok_shuffle")  # get state dict
    """

    def __init__(self, root: Path):
        self.root = Path(root)
        self._entries: Dict[str, LoRAEntry] = {}
        self._lock = threading.Lock()
        self.root.mkdir(parents=True, exist_ok=True)

    # ──────────────────────────────────────────────────────────────────────────
    # Discovery
    # ──────────────────────────────────────────────────────────────────────────

    def scan(self) -> List[str]:
        """
        Scan registry root for category directories containing lora_weights.safetensors.
        Returns list of discovered category names.
        """
        found: List[str] = []
        with self._lock:
            for subdir in sorted(self.root.iterdir()):
                if not subdir.is_dir():
                    continue
                weights = subdir / "lora_weights.safetensors"
                if not weights.exists():
                    continue
                category = subdir.name
                meta = self._load_metadata(subdir)
                entry = LoRAEntry(
                    category=category,
                    display_name=CATEGORY_DISPLAY_NAMES.get(category, category.replace("_", " ").title()),
                    weights_path=weights,
                    metadata=meta,
                )
                self._entries[category] = entry
                found.append(category)

        if found:
            logger.info("[Registry] Found %d LoRA(s): %s", len(found), found)
        else:
            logger.info("[Registry] No LoRA weights found in %s", self.root)

        return found

    @staticmethod
    def _load_metadata(subdir: Path) -> Dict[str, Any]:
        meta_path = subdir / "metadata.json"
        if meta_path.exists():
            try:
                return json.loads(meta_path.read_text())
            except Exception as e:
                logger.warning("Failed to parse metadata %s: %s", meta_path, e)
        return {}

    # ──────────────────────────────────────────────────────────────────────────
    # Registration
    # ──────────────────────────────────────────────────────────────────────────

    def register(
        self,
        category: str,
        weights_path: Path,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> LoRAEntry:
        """
        Manually register a LoRA entry (e.g. after training completes).
        Overwrites any existing entry for the same category.
        """
        with self._lock:
            entry = LoRAEntry(
                category=category,
                display_name=CATEGORY_DISPLAY_NAMES.get(category, category.replace("_", " ").title()),
                weights_path=Path(weights_path),
                metadata=metadata or {},
            )
            # Invalidate loaded state if replaced
            old = self._entries.get(category)
            if old and old.is_loaded:
                old.state_dict = None
                old.is_loaded = False

            self._entries[category] = entry
            logger.info("[Registry] Registered LoRA '%s' → %s", category, weights_path)
            return entry

    # ──────────────────────────────────────────────────────────────────────────
    # Access
    # ──────────────────────────────────────────────────────────────────────────

    def get(self, category: str) -> Optional[LoRAEntry]:
        """Return entry for a category, or None if not registered."""
        return self._entries.get(category)

    def list_categories(self) -> List[str]:
        """Return all registered category names."""
        with self._lock:
            return list(self._entries.keys())

    def info(self) -> RegistryInfo:
        """Return serializable registry summary."""
        with self._lock:
            entries = [
                {
                    "category": e.category,
                    "display_name": e.display_name,
                    "is_loaded": e.is_loaded,
                    "lora_rank": e.lora_rank,
                    "base_model": e.base_model,
                    "weights_path": str(e.weights_path),
                }
                for e in self._entries.values()
            ]
            return RegistryInfo(
                total=len(self._entries),
                categories=list(self._entries.keys()),
                loaded=[k for k, e in self._entries.items() if e.is_loaded],
                entries=entries,
            )

    # ──────────────────────────────────────────────────────────────────────────
    # Weight management
    # ──────────────────────────────────────────────────────────────────────────

    def load_weights(
        self,
        category: str,
        device: str = "cpu",
        dtype: torch.dtype = torch.bfloat16,
    ) -> Optional[Dict[str, torch.Tensor]]:
        """
        Load (and cache) LoRA state dict for a category.
        Returns None if category is not registered.
        """
        entry = self._entries.get(category)
        if entry is None:
            logger.warning("[Registry] Unknown category: '%s'", category)
            return None

        with self._lock:
            if entry.is_loaded and entry.state_dict is not None:
                return entry.state_dict

            if not entry.weights_path.exists():
                logger.error("[Registry] Weights file missing: %s", entry.weights_path)
                return None

            try:
                from safetensors.torch import load_file
                sd = load_file(str(entry.weights_path), device=device)
                sd = {k: v.to(dtype) for k, v in sd.items()}
                entry.state_dict = sd
                entry.is_loaded = True
                logger.info(
                    "[Registry] Loaded '%s' LoRA (%d params, %.1f MB)",
                    category,
                    len(sd),
                    sum(v.nbytes for v in sd.values()) / 1e6,
                )
                return sd
            except Exception as e:
                logger.error("[Registry] Failed to load weights for '%s': %s", category, e)
                return None

    def unload_weights(self, category: str) -> None:
        """Release cached LoRA weights from memory."""
        with self._lock:
            entry = self._entries.get(category)
            if entry and entry.is_loaded:
                entry.state_dict = None
                entry.is_loaded = False
                logger.info("[Registry] Unloaded '%s'", category)

    def unload_all(self) -> None:
        """Release all cached LoRA weights."""
        with self._lock:
            for entry in self._entries.values():
                entry.state_dict = None
                entry.is_loaded = False
        logger.info("[Registry] All LoRAs unloaded")

    # ──────────────────────────────────────────────────────────────────────────
    # Prompt helpers
    # ──────────────────────────────────────────────────────────────────────────

    def get_prompt(self, category: str, character: str = "dog") -> str:
        """Return a generation prompt for the given category and character."""
        template = CATEGORY_PROMPTS.get(
            category,
            "anthropomorphic {character} performing {category} dance, "
            "full body visible, dynamic motion, high quality",
        )
        return template.format(character=character, category=category.replace("_", " "))

    @staticmethod
    def get_negative_prompt() -> str:
        return NEGATIVE_PROMPT


# ──────────────────────────────────────────────────────────────────────────────
# Singleton
# ──────────────────────────────────────────────────────────────────────────────

_registry_instance: Optional[LoRARegistry] = None
_registry_lock = threading.Lock()


def get_registry(root: Optional[Path] = None) -> LoRARegistry:
    """
    Return the global LoRARegistry singleton.
    Auto-scans on first access.
    """
    global _registry_instance
    with _registry_lock:
        if _registry_instance is None:
            if root is None:
                from app.core.config import get_settings
                settings = get_settings()
                root = settings.backend_dir / "data" / "reels_lora"
            _registry_instance = LoRARegistry(root=root)
            _registry_instance.scan()
    return _registry_instance
