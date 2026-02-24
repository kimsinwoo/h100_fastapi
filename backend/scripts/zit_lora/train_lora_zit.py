"""
Z-Image-Turbo (S3-DiT) LoRA 학습.
- HF 데이터셋만 직접 로드(로컬 export 금지). resize + center crop만.
- target_modules 자동 탐색, rank/epochs 스타일별, lr=1e-4, adamw_8bit, bfloat16, torch.compile.
- 출력: lora_output/{style}.safetensors
"""
from __future__ import annotations

import gc
import logging
import os
import sys
from pathlib import Path
from typing import Any

# scripts/zit_lora 기준 상대 경로
_SCRIPT_DIR = Path(__file__).resolve().parent
if str(_SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(_SCRIPT_DIR))

os.environ.setdefault("PYTORCH_JIT", "0")

# torch / torchvision 호환성 검사 (transformers가 torchvision을 쓰므로 먼저 검사)
def _check_torch_torchvision() -> None:
    import torch
    try:
        import torchvision  # noqa: F401
    except RuntimeError as e:
        if "torchvision::nms" in str(e) or "does not exist" in str(e):
            print(
                "ERROR: torch와 torchvision 버전이 맞지 않습니다.\n"
                "다음 중 하나로 맞춰 설치한 뒤 다시 실행하세요:\n"
                "  pip install torch torchvision --upgrade\n"
                "또는 PyTorch 공식에 맞춰 (예: CUDA 12.1):\n"
                "  pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121\n",
                file=sys.stderr,
            )
            sys.exit(1)
        raise

_check_torch_torchvision()

from config import (
    STYLE_CONFIGS,
    STYLE_DATASET,
    RESOLUTION,
    ZIT_MODEL_ID,
    LEARNING_RATE,
    OPTIMIZER,
    WEIGHT_DECAY,
    LORA_OUTPUT_DIR,
    StyleConfig,
)
from dataset_hf import HFDatasetWrapper

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(name)s: %(message)s")
logger = logging.getLogger(__name__)

import torch
from torch.utils.data import DataLoader


def _pil_to_tensor(pil_img: Any) -> torch.Tensor:
    """PIL -> [0,1] -> [-1,1] (torchvision 미사용, 호환성 회피)."""
    import numpy as np
    arr = np.array(pil_img, dtype=np.float32) / 255.0
    arr = (arr - 0.5) / 0.5
    return torch.from_numpy(arr).permute(2, 0, 1)


def _get_transformer(pipe: Any) -> Any:
    if hasattr(pipe, "transformer"):
        return pipe.transformer
    if hasattr(pipe, "dit"):
        return pipe.dit
    raise AttributeError("Pipeline has no 'transformer' or 'dit'.")


def _discover_target_modules(model: Any) -> list[str]:
    import torch.nn as nn
    patterns = ("to_q", "to_k", "to_v", "to_out.0", "w1", "w2", "w3")
    found: set[str] = set()
    for name, module in model.named_modules():
        if not isinstance(module, nn.Linear):
            continue
        for p in patterns:
            if p in name or name.endswith("." + p):
                found.add(name)
                break
    return sorted(found)


def _load_pipeline(device: str, dtype: torch.dtype) -> Any:
    try:
        import diffusers.utils.import_utils as du
        du.is_xformers_available = lambda: False
    except Exception:
        pass
    from diffusers import ZImageImg2ImgPipeline
    pipe = ZImageImg2ImgPipeline.from_pretrained(
        ZIT_MODEL_ID,
        torch_dtype=dtype,
        variant="fp16" if dtype == torch.float16 else None,
    )
    return pipe.to(device)


def _apply_lora(transformer: Any, target_modules: list[str], rank: int, alpha: int) -> Any:
    from peft import LoraConfig, get_peft_model, TaskType
    if not target_modules:
        raise ValueError("target_modules empty")
    cfg = LoraConfig(
        r=rank,
        lora_alpha=alpha,
        target_modules=target_modules,
        lora_dropout=0.0,
        bias="none",
        task_type=TaskType.FEATURE_EXTRACTION,
        inference_mode=False,
    )
    return get_peft_model(transformer, cfg)


def _collate(batch: list[dict[str, Any]]) -> dict[str, Any]:
    pixel_values = torch.stack([_pil_to_tensor(b["image"]) for b in batch])
    captions = [b["caption"] for b in batch]
    return {"pixel_values": pixel_values, "captions": captions}


def _train_one_epoch(
    transformer_lora: Any,
    vae: Any,
    text_encoder: Any,
    tokenizer: Any,
    scheduler: Any,
    dataloader: DataLoader[dict[str, Any]],
    optimizer: torch.optim.Optimizer,
    scaler: torch.amp.GradScaler,
    device: str,
    dtype: torch.dtype,
    resolution: int,
    grad_accum: int,
) -> float:
    transformer_lora.train()
    vae.eval()
    text_encoder.eval()
    total_loss = 0.0
    n = 0
    for step, batch in enumerate(dataloader):
        pixel_values = batch["pixel_values"].to(device, dtype=dtype)
        captions = batch["captions"]
        with torch.no_grad():
            latents = vae.encode(pixel_values).latent_dist.sample()
            latents = latents * (getattr(vae.config, "scaling_factor", 0.18215) or 0.18215)
            tokens = tokenizer(
                captions,
                padding="max_length",
                max_length=77,
                truncation=True,
                return_tensors="pt",
            ).to(device)
            enc = text_encoder(tokens.input_ids)[0]
            noise = torch.randn_like(latents, device=device, dtype=dtype)
            bsz = latents.shape[0]
            # FlowMatchEulerDiscreteScheduler uses scale_noise; timestep must match scheduler.timesteps
            if hasattr(scheduler, "scale_noise") and hasattr(scheduler, "timesteps"):
                timesteps = scheduler.timesteps.to(device=device, dtype=latents.dtype)
                idx = torch.randint(0, len(timesteps), (bsz,), device=device)
                t = timesteps[idx]
                noisy = scheduler.scale_noise(latents, t, noise)
            else:
                t = torch.randint(0, scheduler.config.num_train_timesteps, (bsz,), device=device).long()
                noisy = scheduler.add_noise(latents, noise, t)
        out = transformer_lora(noisy, t, encoder_hidden_states=enc, return_dict=True)
        pred = out.sample if hasattr(out, "sample") else out
        loss = torch.nn.functional.mse_loss(pred.float(), noise.float(), reduction="mean") / grad_accum
        scaler.scale(loss).backward()
        if (step + 1) % grad_accum == 0:
            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad()
        total_loss += loss.item() * grad_accum
        n += 1
    return total_loss / max(n, 1)


def run_training(style_key: str, batch_size: int = 1, grad_accum: int = 1) -> Path:
    if style_key not in STYLE_CONFIGS or style_key not in STYLE_DATASET:
        raise ValueError(f"Unknown style_key: {style_key}. Allowed: {list(STYLE_CONFIGS.keys())}")
    cfg = STYLE_CONFIGS[style_key]
    LORA_OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    out_file = LORA_OUTPUT_DIR / f"{style_key}.safetensors"

    device = "cuda" if torch.cuda.is_available() else "cpu"
    dtype = torch.bfloat16 if (device == "cuda" and torch.cuda.is_bf16_supported()) else torch.float16

    pipe = _load_pipeline(device, dtype)
    transformer = _get_transformer(pipe)
    vae = pipe.vae
    text_encoder = pipe.text_encoder
    tokenizer = pipe.tokenizer
    noise_scheduler = pipe.scheduler

    if hasattr(transformer, "gradient_checkpointing_enable"):
        transformer.gradient_checkpointing_enable()
    target_modules = _discover_target_modules(transformer)
    logger.info("LoRA target_modules (%d): %s", len(target_modules), target_modules[:6])
    transformer_lora = _apply_lora(transformer, target_modules, cfg.rank, cfg.alpha)
    if device == "cuda" and os.environ.get("TORCH_COMPILE_DISABLE") != "1":
        try:
            transformer_lora = torch.compile(transformer_lora, mode="reduce-overhead")
        except Exception as e:
            logger.warning("torch.compile skipped: %s", e)

    dataset = HFDatasetWrapper(style_key, resolution=RESOLUTION)
    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=0,
        collate_fn=_collate,
    )

    if OPTIMIZER == "adamw_8bit":
        try:
            import bitsandbytes
            optimizer = bitsandbytes.optim.AdamW8bit(
                transformer_lora.parameters(),
                lr=LEARNING_RATE,
                weight_decay=WEIGHT_DECAY,
            )
        except ImportError:
            optimizer = torch.optim.AdamW(transformer_lora.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY)
    else:
        optimizer = torch.optim.AdamW(transformer_lora.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY)
    scheduler_lr = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=cfg.epochs)
    scaler = torch.amp.GradScaler("cuda", enabled=(device == "cuda" and dtype == torch.float16))

    for epoch in range(cfg.epochs):
        avg = _train_one_epoch(
            transformer_lora, vae, text_encoder, tokenizer, noise_scheduler,
            dataloader, optimizer, scaler, device, dtype, RESOLUTION, grad_accum,
        )
        scheduler_lr.step()
        logger.info("Epoch %d/%d train_loss=%.4f", epoch + 1, cfg.epochs, avg)
        gc.collect()
        if device == "cuda":
            torch.cuda.empty_cache()

    # lora_output/{style}.safetensors 단일 파일로 저장
    import safetensors.torch as st
    state = transformer_lora.state_dict()
    st.save_file(state, out_file)
    logger.info("Saved: %s", out_file)
    return out_file


def main() -> int:
    import argparse
    p = argparse.ArgumentParser()
    p.add_argument("--style", type=str, required=True, choices=list(STYLE_CONFIGS.keys()))
    p.add_argument("--batch_size", type=int, default=1)
    p.add_argument("--gradient_accumulation_steps", type=int, default=1)
    args = p.parse_args()
    try:
        run_training(args.style, batch_size=args.batch_size, grad_accum=args.gradient_accumulation_steps)
        return 0
    except Exception as e:
        logger.exception("%s", e)
        return 1


if __name__ == "__main__":
    sys.exit(main())
