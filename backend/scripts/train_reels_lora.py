#!/usr/bin/env python3
"""
CLI training script for Reels Dance Motion LoRA.

Single GPU:
    python scripts/train_reels_lora.py \
        --category tiktok_shuffle \
        --video_dir data/reels_raw \
        --num_train_steps 5000

Multi-GPU (accelerate):
    accelerate launch --num_processes 4 scripts/train_reels_lora.py \
        --category tiktok_shuffle \
        --video_dir data/reels_raw \
        --num_train_steps 10000 \
        --batch_size 4

Use YAML config:
    python scripts/train_reels_lora.py --config configs/reels_lora/base.yaml \
        --category kpop_challenge
"""

from __future__ import annotations

import argparse
import logging
import os
import sys
from pathlib import Path

# Add backend to sys.path so app imports work
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import yaml

from app.services.reels_lora.lora_trainer import LoRATrainingConfig, ReelsLoRATrainer
from app.services.reels_lora.lora_registry import get_registry

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)],
)
logger = logging.getLogger(__name__)


# ──────────────────────────────────────────────────────────────────────────────
# Argument parser
# ──────────────────────────────────────────────────────────────────────────────

def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="Train Reels Dance Motion LoRA for AnimateDiff",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    # Config file (overrides defaults; CLI args override config file)
    p.add_argument("--config", type=str, default=None, help="YAML config file path")

    # Model
    p.add_argument("--base_model", type=str, default="runwayml/stable-diffusion-v1-5")
    p.add_argument("--motion_adapter", type=str, default="guoyww/animatediff-motion-adapter-v1-5-2")

    # LoRA
    p.add_argument("--lora_rank", type=int, default=32)
    p.add_argument("--lora_alpha", type=int, default=32)
    p.add_argument("--lora_dropout", type=float, default=0.05)

    # Data
    p.add_argument("--category", type=str, required=False,
                   help="Dance category name (required if not in config)")
    p.add_argument("--video_dir", type=str, default="data/reels_raw")
    p.add_argument("--clip_frames", type=int, default=16)
    p.add_argument("--image_size", type=int, default=512)
    p.add_argument("--batch_size", type=int, default=2)
    p.add_argument("--num_workers", type=int, default=4)
    p.add_argument("--cache_dir", type=str, default="data/reels_cache")

    # Training
    p.add_argument("--num_train_steps", type=int, default=5000)
    p.add_argument("--gradient_accumulation_steps", type=int, default=4)
    p.add_argument("--learning_rate", type=float, default=1e-4)
    p.add_argument("--lr_warmup_steps", type=int, default=200)
    p.add_argument("--weight_decay", type=float, default=1e-2)
    p.add_argument("--max_grad_norm", type=float, default=1.0)
    p.add_argument("--mixed_precision", choices=["bf16", "fp16", "no"], default="bf16")
    p.add_argument("--no_gradient_checkpointing", action="store_true")

    # H100 features
    p.add_argument("--enable_tf32", action="store_true", default=True)
    p.add_argument("--enable_torch_compile", action="store_true", default=False)
    p.add_argument("--no_flash_attention", action="store_true")

    # Output
    p.add_argument("--output_dir", type=str, default="data/reels_lora")
    p.add_argument("--save_every", type=int, default=500)
    p.add_argument("--log_every", type=int, default=50)

    return p


# ──────────────────────────────────────────────────────────────────────────────
# Config loader
# ──────────────────────────────────────────────────────────────────────────────

def load_yaml_config(path: str) -> dict:
    with open(path) as f:
        return yaml.safe_load(f) or {}


def merge_config(yaml_cfg: dict, args: argparse.Namespace) -> dict:
    """Merge: YAML defaults → CLI overrides."""
    merged = dict(yaml_cfg)
    cli = vars(args)
    # CLI args that were explicitly set take priority
    for k, v in cli.items():
        if k == "config":
            continue
        if v is not None:
            merged[k] = v
    return merged


# ──────────────────────────────────────────────────────────────────────────────
# Main
# ──────────────────────────────────────────────────────────────────────────────

def main() -> None:
    parser = build_parser()
    args = parser.parse_args()

    # Load YAML config if provided
    yaml_cfg: dict = {}
    if args.config:
        logger.info("Loading config: %s", args.config)
        yaml_cfg = load_yaml_config(args.config)

    merged = merge_config(yaml_cfg, args)

    # Validate category
    category = merged.get("category")
    if not category:
        parser.error("--category is required (either via CLI or YAML config)")

    # Build training config
    cfg = LoRATrainingConfig(
        base_model_id=merged.get("base_model", "runwayml/stable-diffusion-v1-5"),
        motion_adapter_id=merged.get("motion_adapter", "guoyww/animatediff-motion-adapter-v1-5-2"),
        lora_rank=int(merged.get("lora_rank", 32)),
        lora_alpha=int(merged.get("lora_alpha", 32)),
        lora_dropout=float(merged.get("lora_dropout", 0.05)),
        category=category,
        video_dir=str(merged.get("video_dir", "data/reels_raw")),
        clip_frames=int(merged.get("clip_frames", 16)),
        image_size=int(merged.get("image_size", 512)),
        batch_size=int(merged.get("batch_size", 2)),
        num_workers=int(merged.get("num_workers", 4)),
        cache_dir=str(merged.get("cache_dir", "data/reels_cache")),
        num_train_steps=int(merged.get("num_train_steps", 5000)),
        gradient_accumulation_steps=int(merged.get("gradient_accumulation_steps", 4)),
        learning_rate=float(merged.get("learning_rate", 1e-4)),
        lr_warmup_steps=int(merged.get("lr_warmup_steps", 200)),
        weight_decay=float(merged.get("weight_decay", 1e-2)),
        max_grad_norm=float(merged.get("max_grad_norm", 1.0)),
        mixed_precision=str(merged.get("mixed_precision", "bf16")),
        gradient_checkpointing=not bool(merged.get("no_gradient_checkpointing", False)),
        output_dir=str(merged.get("output_dir", "data/reels_lora")),
        save_every=int(merged.get("save_every", 500)),
        log_every=int(merged.get("log_every", 50)),
        enable_tf32=bool(merged.get("enable_tf32", True)),
        enable_torch_compile=bool(merged.get("enable_torch_compile", False)),
        enable_flash_attention=not bool(merged.get("no_flash_attention", False)),
    )

    # Pre-flight checks
    video_category_dir = Path(cfg.video_dir) / cfg.category
    if not video_category_dir.exists():
        logger.error("Video directory not found: %s", video_category_dir)
        logger.error("Create it and add dance videos (mp4/mov) before training.")
        sys.exit(1)

    video_files = [
        p for p in video_category_dir.iterdir()
        if p.suffix.lower() in {".mp4", ".mov", ".avi", ".webm"}
    ]
    if not video_files:
        logger.error("No video files found in %s", video_category_dir)
        sys.exit(1)

    logger.info("=" * 60)
    logger.info("Reels Dance LoRA Training")
    logger.info("  Category     : %s", cfg.category)
    logger.info("  Videos found : %d", len(video_files))
    logger.info("  Train steps  : %d", cfg.num_train_steps)
    logger.info("  LoRA rank    : %d", cfg.lora_rank)
    logger.info("  Image size   : %d", cfg.image_size)
    logger.info("  Batch size   : %d (× grad_accum %d = eff. %d)",
                cfg.batch_size, cfg.gradient_accumulation_steps,
                cfg.batch_size * cfg.gradient_accumulation_steps)
    logger.info("  Mixed prec.  : %s", cfg.mixed_precision)
    logger.info("  Output dir   : %s", cfg.output_dir)
    logger.info("=" * 60)

    # Train
    trainer = ReelsLoRATrainer(cfg)
    output_path = trainer.train()

    # Register weights in the global registry
    registry = get_registry(root=Path(cfg.output_dir))
    registry.register(cfg.category, output_path, metadata={"train_steps": cfg.num_train_steps})

    logger.info("=" * 60)
    logger.info("Training complete!")
    logger.info("  Weights saved → %s", output_path)
    logger.info("  Registry updated for category '%s'", cfg.category)
    logger.info("=" * 60)


if __name__ == "__main__":
    main()
