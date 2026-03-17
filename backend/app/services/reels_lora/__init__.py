"""
Reels Dance Motion LoRA System
===============================
Workflow:
  1. Dance videos  →  dataset_pipeline   : frame extraction + DW Pose skeleton
  2. Skeletons     →  motion_dataset     : temporal normalization, PyTorch Dataset
  3. Dataset       →  lora_trainer       : AnimateDiff LoRA training (bf16, H100)
  4. Weights       →  lora_registry      : category-based storage & lookup
  5. Inference     →  inference_extension: dynamic LoRA injection into video pipeline
"""

from .lora_registry import LoRARegistry, get_registry
from .inference_extension import ReelsDanceGenerator

__all__ = ["LoRARegistry", "get_registry", "ReelsDanceGenerator"]
