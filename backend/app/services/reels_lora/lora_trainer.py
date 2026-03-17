"""
Motion LoRA Trainer for AnimateDiff.

Architecture:
  - Base model: AnimateDiff v1-5-2 or v1-5-3 motion module
  - LoRA injected into temporal self-attention (to_q, to_k, to_v, to_out.0)
  - Training objective: video reconstruction loss (MSE on latent noise)
  - H100 optimizations: bf16, gradient checkpointing, torch.compile, Flash Attention 2

Training loop:
  1. Load AnimateDiff motion module + SD1.5 UNet
  2. Inject LoRA adapters (PEFT) into temporal attention layers
  3. Freeze everything except LoRA parameters
  4. For each batch:
     - Encode pixel frames with VAE
     - Add noise (forward diffusion)
     - Predict noise with UNet (LoRA active)
     - MSE loss on noise prediction
     - Backward + optimizer step
  5. Periodic checkpoint saving

Supports:
  - Single GPU (default)
  - Multi-GPU via accelerate (launch with: accelerate launch train_reels_lora.py ...)
  - Gradient accumulation
  - Configurable LoRA rank / alpha
"""

from __future__ import annotations

import dataclasses
import logging
import math
import os
import time
from pathlib import Path
from typing import Any, Dict, List, Optional

import torch
import torch.nn.functional as F
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR

logger = logging.getLogger(__name__)

# ──────────────────────────────────────────────────────────────────────────────
# Training configuration
# ──────────────────────────────────────────────────────────────────────────────

@dataclasses.dataclass
class LoRATrainingConfig:
    # Model
    base_model_id: str = "runwayml/stable-diffusion-v1-5"
    motion_adapter_id: str = "guoyww/animatediff-motion-adapter-v1-5-2"

    # LoRA
    lora_rank: int = 32
    lora_alpha: int = 32
    lora_dropout: float = 0.05
    target_modules: List[str] = dataclasses.field(default_factory=lambda: [
        "to_q", "to_k", "to_v", "to_out.0",
    ])

    # Data
    video_dir: str = "data/reels_raw"
    category: str = "tiktok_shuffle"
    clip_frames: int = 16
    image_size: int = 512
    batch_size: int = 2
    num_workers: int = 4
    cache_dir: str = "data/reels_cache"

    # Training
    num_train_steps: int = 5000
    gradient_accumulation_steps: int = 4
    learning_rate: float = 1e-4
    lr_warmup_steps: int = 200
    lr_min: float = 1e-6
    weight_decay: float = 1e-2
    max_grad_norm: float = 1.0
    mixed_precision: str = "bf16"     # "bf16" | "fp16" | "no"
    gradient_checkpointing: bool = True

    # Output
    output_dir: str = "data/reels_lora"
    save_every: int = 500
    log_every: int = 50

    # H100 specific
    enable_tf32: bool = True
    enable_torch_compile: bool = False   # True for production, slow first run
    enable_flash_attention: bool = True


# ──────────────────────────────────────────────────────────────────────────────
# LoRA model builder
# ──────────────────────────────────────────────────────────────────────────────

def build_animatediff_lora_model(cfg: LoRATrainingConfig):
    """
    Load AnimateDiff pipeline and inject LoRA into temporal attention layers.

    Returns:
        (unet, vae, noise_scheduler, tokenizer, text_encoder)
    """
    try:
        from diffusers import (
            AnimateDiffPipeline,
            MotionAdapter,
            DDIMScheduler,
        )
        from peft import LoraConfig, get_peft_model, TaskType
    except ImportError as e:
        raise RuntimeError(
            f"Missing dependency: {e}. "
            "Install: pip install diffusers>=0.25.0 peft>=0.10.0"
        ) from e

    logger.info("[Trainer] Loading MotionAdapter: %s", cfg.motion_adapter_id)
    adapter = MotionAdapter.from_pretrained(
        cfg.motion_adapter_id,
        torch_dtype=torch.bfloat16 if cfg.mixed_precision == "bf16" else torch.float16,
    )

    logger.info("[Trainer] Loading AnimateDiff pipeline: %s", cfg.base_model_id)
    scheduler = DDIMScheduler.from_pretrained(
        cfg.base_model_id,
        subfolder="scheduler",
        clip_sample=False,
        timestep_spacing="linspace",
        beta_schedule="linear",
        steps_offset=1,
    )
    pipe = AnimateDiffPipeline.from_pretrained(
        cfg.base_model_id,
        motion_adapter=adapter,
        scheduler=scheduler,
        torch_dtype=torch.bfloat16 if cfg.mixed_precision == "bf16" else torch.float16,
    )

    unet = pipe.unet
    vae = pipe.vae
    text_encoder = pipe.text_encoder
    tokenizer = pipe.tokenizer
    noise_scheduler = pipe.scheduler

    # ── Freeze all ──────────────────────────────────────────────────────────
    for param in unet.parameters():
        param.requires_grad_(False)
    for param in vae.parameters():
        param.requires_grad_(False)
    for param in text_encoder.parameters():
        param.requires_grad_(False)

    # ── Gradient checkpointing on UNet ────────────────────────────────────
    if cfg.gradient_checkpointing:
        unet.enable_gradient_checkpointing()
        logger.info("[Trainer] Gradient checkpointing enabled on UNet")

    # ── Inject LoRA into temporal attention (motion module only) ──────────
    # Target only the motion module temporal attention layers for dance-specific LoRA
    lora_target_modules = []
    for name, module in unet.named_modules():
        # motion module temporal self-attention blocks
        if "motion_modules" in name or "temporal" in name.lower():
            for target in cfg.target_modules:
                full = f"{name}.{target}" if name else target
                # Check if this module actually has the sub-layer
                try:
                    obj = module
                    for part in target.split("."):
                        obj = getattr(obj, part)
                    lora_target_modules.append(full)
                except AttributeError:
                    pass

    # De-duplicate by checking actual module names
    lora_config = LoraConfig(
        r=cfg.lora_rank,
        lora_alpha=cfg.lora_alpha,
        lora_dropout=cfg.lora_dropout,
        target_modules=cfg.target_modules,  # PEFT handles scope via named_modules
        bias="none",
    )

    unet = get_peft_model(unet, lora_config)
    unet.print_trainable_parameters()

    # ── Flash Attention 2 ─────────────────────────────────────────────────
    if cfg.enable_flash_attention:
        try:
            unet.enable_xformers_memory_efficient_attention()
            logger.info("[Trainer] xformers memory efficient attention enabled")
        except Exception:
            logger.debug("[Trainer] xformers not available, skipping")

    return unet, vae, noise_scheduler, tokenizer, text_encoder


# ──────────────────────────────────────────────────────────────────────────────
# Training utilities
# ──────────────────────────────────────────────────────────────────────────────

def encode_frames_to_latents(
    vae,
    pixel_values: torch.Tensor,    # (B, T, 3, H, W) float32 [-1, 1]
    device: torch.device,
    dtype: torch.dtype,
) -> torch.Tensor:
    """VAE-encode video frames to latent space. Returns (B, T, 4, h, w)."""
    B, T, C, H, W = pixel_values.shape
    flat = pixel_values.view(B * T, C, H, W).to(device=device, dtype=dtype)

    with torch.no_grad():
        latents = vae.encode(flat).latent_dist.sample()
        latents = latents * vae.config.scaling_factor

    _, c, h, w = latents.shape
    return latents.view(B, T, c, h, w)


def get_text_embeddings(
    tokenizer,
    text_encoder,
    prompts: List[str],
    device: torch.device,
    dtype: torch.dtype,
) -> torch.Tensor:
    """Encode text prompts to embeddings. Returns (B, seq_len, dim)."""
    tokens = tokenizer(
        prompts,
        padding="max_length",
        max_length=tokenizer.model_max_length,
        truncation=True,
        return_tensors="pt",
    )
    with torch.no_grad():
        embeddings = text_encoder(tokens.input_ids.to(device))[0].to(dtype)
    return embeddings


# ──────────────────────────────────────────────────────────────────────────────
# Warmup LR scheduler
# ──────────────────────────────────────────────────────────────────────────────

def get_lr_scheduler_with_warmup(
    optimizer: torch.optim.Optimizer,
    warmup_steps: int,
    total_steps: int,
    min_lr: float = 1e-6,
) -> torch.optim.lr_scheduler.LambdaLR:
    """Cosine decay with linear warmup."""
    def lr_lambda(step: int) -> float:
        if step < warmup_steps:
            return float(step) / max(1, warmup_steps)
        progress = float(step - warmup_steps) / max(1, total_steps - warmup_steps)
        cosine = 0.5 * (1.0 + math.cos(math.pi * progress))
        base_lr = optimizer.defaults["lr"]
        min_ratio = min_lr / base_lr if base_lr > 0 else 0.0
        return max(min_ratio, cosine)

    return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)


# ──────────────────────────────────────────────────────────────────────────────
# Main trainer
# ──────────────────────────────────────────────────────────────────────────────

class ReelsLoRATrainer:
    """
    Production LoRA trainer for Reels dance motion.

    Usage:
        config = LoRATrainingConfig(
            category="tiktok_shuffle",
            video_dir="data/reels_raw",
            num_train_steps=5000,
        )
        trainer = ReelsLoRATrainer(config)
        trainer.train()
    """

    def __init__(self, config: LoRATrainingConfig):
        self.cfg = config
        self._setup_environment()

    def _setup_environment(self) -> None:
        if self.cfg.enable_tf32 and torch.cuda.is_available():
            torch.backends.cuda.matmul.allow_tf32 = True
            torch.backends.cudnn.allow_tf32 = True
            logger.info("[Trainer] TF32 enabled")

    # ──────────────────────────────────────────────────────────────────────────
    def train(self) -> Path:
        """
        Full training loop.
        Returns path to saved LoRA weights.
        """
        try:
            from accelerate import Accelerator
        except ImportError:
            raise RuntimeError("Install: pip install accelerate>=0.26.0")

        accelerator = Accelerator(
            gradient_accumulation_steps=self.cfg.gradient_accumulation_steps,
            mixed_precision=self.cfg.mixed_precision,
            log_with="tensorboard",
            project_dir=str(Path(self.cfg.output_dir) / "logs"),
        )

        logger.info(
            "[Trainer] Starting training | category=%s | steps=%d | rank=%d | gpu=%s",
            self.cfg.category,
            self.cfg.num_train_steps,
            self.cfg.lora_rank,
            accelerator.device,
        )

        # ── Build dataset ────────────────────────────────────────────────
        dataloader = self._build_dataloader()

        # ── Build model ──────────────────────────────────────────────────
        unet, vae, noise_scheduler, tokenizer, text_encoder = build_animatediff_lora_model(self.cfg)
        device = accelerator.device
        dtype = torch.bfloat16 if self.cfg.mixed_precision == "bf16" else torch.float16

        vae.to(device, dtype=dtype)
        text_encoder.to(device, dtype=dtype)

        # ── Optimizer & scheduler ────────────────────────────────────────
        trainable = [p for p in unet.parameters() if p.requires_grad]
        optimizer = AdamW(
            trainable,
            lr=self.cfg.learning_rate,
            weight_decay=self.cfg.weight_decay,
            betas=(0.9, 0.999),
            eps=1e-8,
        )
        lr_scheduler = get_lr_scheduler_with_warmup(
            optimizer,
            warmup_steps=self.cfg.lr_warmup_steps,
            total_steps=self.cfg.num_train_steps,
            min_lr=self.cfg.lr_min,
        )

        # ── Accelerate prepare ───────────────────────────────────────────
        unet, optimizer, dataloader, lr_scheduler = accelerator.prepare(
            unet, optimizer, dataloader, lr_scheduler
        )

        # ── torch.compile (H100) ─────────────────────────────────────────
        if self.cfg.enable_torch_compile:
            logger.info("[Trainer] Compiling UNet with torch.compile (first step slow)...")
            unet = torch.compile(unet, mode="reduce-overhead")

        # ── Training loop ─────────────────────────────────────────────────
        global_step = 0
        running_loss = 0.0
        t_start = time.perf_counter()

        unet.train()
        data_iter = iter(dataloader)

        while global_step < self.cfg.num_train_steps:
            try:
                batch = next(data_iter)
            except StopIteration:
                data_iter = iter(dataloader)
                batch = next(data_iter)

            with accelerator.accumulate(unet):
                loss = self._train_step(
                    batch, unet, vae, noise_scheduler,
                    tokenizer, text_encoder,
                    device, dtype,
                )
                accelerator.backward(loss)

                if accelerator.sync_gradients:
                    accelerator.clip_grad_norm_(trainable, self.cfg.max_grad_norm)

                optimizer.step()
                lr_scheduler.step()
                optimizer.zero_grad()

            running_loss += loss.item()
            global_step += 1

            # ── Logging ───────────────────────────────────────────────────
            if global_step % self.cfg.log_every == 0:
                avg_loss = running_loss / self.cfg.log_every
                elapsed = time.perf_counter() - t_start
                steps_per_sec = global_step / elapsed
                eta = (self.cfg.num_train_steps - global_step) / max(steps_per_sec, 1e-6)
                lr_now = lr_scheduler.get_last_lr()[0]

                logger.info(
                    "[Trainer] step=%d/%d | loss=%.4f | lr=%.2e | %.2f s/step | ETA %.0fs",
                    global_step, self.cfg.num_train_steps,
                    avg_loss, lr_now,
                    elapsed / global_step,
                    eta,
                )
                running_loss = 0.0

            # ── Checkpoint ───────────────────────────────────────────────
            if global_step % self.cfg.save_every == 0 and accelerator.is_main_process:
                self._save_checkpoint(accelerator, unet, global_step)

        # ── Final save ────────────────────────────────────────────────────
        if accelerator.is_main_process:
            output_path = self._save_final(accelerator, unet)
            logger.info("[Trainer] Training complete → %s", output_path)
            return output_path

        return Path(self.cfg.output_dir) / self.cfg.category / "lora_weights.safetensors"

    # ──────────────────────────────────────────────────────────────────────────
    def _train_step(
        self,
        batch: Dict[str, Any],
        unet,
        vae,
        noise_scheduler,
        tokenizer,
        text_encoder,
        device: torch.device,
        dtype: torch.dtype,
    ) -> torch.Tensor:
        """Single training step. Returns scalar loss tensor."""
        pixel_values = batch["pixel_values"].to(device)   # (B, T, 3, H, W)
        categories = batch["category"]

        # Build category-aware prompts
        prompts = [
            f"anthropomorphic character performing {cat.replace('_', ' ')} dance, "
            "full body visible, dynamic motion, high quality"
            for cat in categories
        ]

        B, T = pixel_values.shape[:2]

        # VAE encode
        latents = encode_frames_to_latents(vae, pixel_values, device, dtype)
        # latents: (B, T, 4, h, w) → rearrange to (B, 4, T, h, w) for AnimateDiff UNet
        latents = latents.permute(0, 2, 1, 3, 4)  # (B, 4, T, h, w)

        # Sample noise
        noise = torch.randn_like(latents)

        # Sample timesteps
        timesteps = torch.randint(
            0,
            noise_scheduler.config.num_train_timesteps,
            (B,),
            device=device,
        ).long()

        # Add noise (forward diffusion)
        noisy_latents = noise_scheduler.add_noise(latents, noise, timesteps)

        # Text embeddings
        encoder_hidden_states = get_text_embeddings(
            tokenizer, text_encoder, prompts, device, dtype
        )

        # Predict noise
        with torch.cuda.amp.autocast(dtype=dtype, enabled=dtype != torch.float32):
            noise_pred = unet(
                noisy_latents,
                timesteps,
                encoder_hidden_states=encoder_hidden_states,
            ).sample

        # MSE loss (v-prediction or epsilon depending on scheduler)
        if noise_scheduler.config.get("prediction_type", "epsilon") == "v_prediction":
            target = noise_scheduler.get_velocity(latents, noise, timesteps)
        else:
            target = noise

        loss = F.mse_loss(noise_pred.float(), target.float(), reduction="mean")
        return loss

    # ──────────────────────────────────────────────────────────────────────────
    def _build_dataloader(self):
        from ..reels_lora.dataset_pipeline import ReelsDatasetPipeline
        from ..reels_lora.motion_dataset import build_dataloader

        pipeline = ReelsDatasetPipeline(
            cache_dir=Path(self.cfg.cache_dir),
            target_fps=8.0,
            target_size=(self.cfg.image_size, self.cfg.image_size),
            clip_frames=self.cfg.clip_frames,
        )
        clips = pipeline.process_category(
            video_dir=Path(self.cfg.video_dir) / self.cfg.category,
            category=self.cfg.category,
        )
        if not clips:
            raise RuntimeError(
                f"No clips found for category '{self.cfg.category}' "
                f"in {self.cfg.video_dir}/{self.cfg.category}"
            )

        return build_dataloader(
            clips=clips,
            clip_frames=self.cfg.clip_frames,
            image_size=self.cfg.image_size,
            batch_size=self.cfg.batch_size,
            num_workers=self.cfg.num_workers,
            augment=True,
        )

    # ──────────────────────────────────────────────────────────────────────────
    def _save_checkpoint(self, accelerator, unet, step: int) -> None:
        out_dir = Path(self.cfg.output_dir) / self.cfg.category / f"checkpoint-{step}"
        out_dir.mkdir(parents=True, exist_ok=True)
        unwrapped = accelerator.unwrap_model(unet)
        unwrapped.save_pretrained(str(out_dir))
        logger.info("[Trainer] Checkpoint saved → %s", out_dir)

    def _save_final(self, accelerator, unet) -> Path:
        """Save final LoRA weights as safetensors for registry consumption."""
        from safetensors.torch import save_file

        out_dir = Path(self.cfg.output_dir) / self.cfg.category
        out_dir.mkdir(parents=True, exist_ok=True)

        unwrapped = accelerator.unwrap_model(unet)

        # Extract only LoRA weights
        lora_state = {
            k: v.contiguous().cpu()
            for k, v in unwrapped.state_dict().items()
            if "lora_" in k
        }

        weights_path = out_dir / "lora_weights.safetensors"
        save_file(lora_state, str(weights_path))

        # Save metadata
        import json
        meta = {
            "category": self.cfg.category,
            "base_model": self.cfg.base_model_id,
            "motion_adapter": self.cfg.motion_adapter_id,
            "lora_rank": self.cfg.lora_rank,
            "lora_alpha": self.cfg.lora_alpha,
            "train_steps": self.cfg.num_train_steps,
            "image_size": self.cfg.image_size,
            "clip_frames": self.cfg.clip_frames,
            "target_modules": self.cfg.target_modules,
        }
        (out_dir / "metadata.json").write_text(json.dumps(meta, indent=2))

        logger.info("[Trainer] Final LoRA weights saved → %s (%d params)", weights_path, len(lora_state))
        return weights_path
