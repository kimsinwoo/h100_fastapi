#!/usr/bin/env python3
"""
Minimal SDXL LoRA training script. Dataset: dataset_path/images/ + captions.txt (filename\tcaption).
Usage: python train_lora_sdxl.py --dataset_path DATA --output_path OUT --rank 4 --max_train_steps 500
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

import torch
from accelerate import Accelerator
from accelerate.utils import ProjectConfiguration, set_seed
from diffusers import StableDiffusionXLPipeline
from diffusers.optimization import get_scheduler
from PIL import Image
from torch.utils.data import Dataset
from tqdm import tqdm


def load_pairs(dataset_path: Path) -> list[tuple[str, str]]:
    captions_file = dataset_path / "captions.txt"
    images_dir = dataset_path / "images"
    pairs = []
    for line in captions_file.read_text(encoding="utf-8", errors="replace").strip().splitlines():
        line = line.strip()
        if not line or "\t" not in line:
            continue
        filename, caption = line.split("\t", 1)
        img_path = images_dir / filename.strip()
        if img_path.exists():
            pairs.append((str(img_path), caption.strip()))
    return pairs


class CaptionDataset(Dataset):
    def __init__(self, pairs: list[tuple[str, str]], size: int):
        self.pairs = pairs
        self.size = size

    def __len__(self) -> int:
        return len(self.pairs)

    def __getitem__(self, i: int) -> dict:
        img_path, caption = self.pairs[i]
        image = Image.open(img_path).convert("RGB")
        image = image.resize((self.size, self.size), Image.Resampling.LANCZOS)
        import numpy as np
        px = torch.from_numpy(np.array(image)).permute(2, 0, 1).float() / 255.0
        return {"pixel_values": px, "caption": caption}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset_path", type=str, required=True)
    ap.add_argument("--output_path", type=str, required=True)
    ap.add_argument("--rank", type=int, default=4)
    ap.add_argument("--learning_rate", type=float, default=1e-4)
    ap.add_argument("--max_train_steps", type=int, default=500)
    ap.add_argument("--train_batch_size", type=int, default=1)
    ap.add_argument("--resolution", type=int, default=1024)
    ap.add_argument("--resume_from", type=str, default=None)
    args = ap.parse_args()

    set_seed(42)
    dataset_path = Path(args.dataset_path)
    output_path = Path(args.output_path)
    output_path.mkdir(parents=True, exist_ok=True)

    pairs = load_pairs(dataset_path)
    if not pairs:
        raise SystemExit("No pairs in captions.txt")

    accelerator = Accelerator(
        gradient_accumulation_steps=4,
        mixed_precision="bf16",
        project_config=ProjectConfiguration(project_dir=str(output_path), logging_dir=str(output_path / "logs")),
    )

    pipe = StableDiffusionXLPipeline.from_pretrained(
        "stabilityai/stable-diffusion-xl-base-1.0",
        torch_dtype=torch.bfloat16,
        use_safetensors=True,
    )
    pipe.set_progress_bar_config(disable=True)
    pipe.to(accelerator.device)

    from peft import LoraConfig, get_peft_model
    lora_config = LoraConfig(
        r=args.rank,
        lora_alpha=args.rank * 2,
        init_lora_weights="gaussian",
        target_modules=["to_k", "to_q", "to_v", "to_out.0"],
    )
    unet = get_peft_model(pipe.unet, lora_config)
    unet.train()

    train_dataset = CaptionDataset(pairs, args.resolution)
    train_dataloader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=args.train_batch_size,
        shuffle=True,
        num_workers=0,
    )
    optimizer = torch.optim.AdamW(unet.parameters(), lr=args.learning_rate)
    lr_scheduler = get_scheduler("constant", optimizer=optimizer, num_warmup_steps=0, num_training_steps=args.max_train_steps)

    unet, optimizer, train_dataloader, lr_scheduler = accelerator.prepare(unet, optimizer, train_dataloader, lr_scheduler)

    global_step = 0
    for step, batch in enumerate(tqdm(train_dataloader, total=min(len(train_dataloader), args.max_train_steps))):
        if global_step >= args.max_train_steps:
            break
        with accelerator.accumulate(unet):
            pixel_values = batch["pixel_values"].to(accelerator.device, dtype=torch.bfloat16)
            captions = batch["caption"]
            if isinstance(captions, torch.Tensor):
                captions = [captions[i].item() if captions[i].numel() == 1 else str(captions[i]) for i in range(len(captions))]
            else:
                captions = list(captions)

            with torch.no_grad():
                latents = pipe.vae.encode(pixel_values).latent_dist.sample() * pipe.vae.config.scaling_factor
            bsz = latents.shape[0]
            timesteps = torch.randint(0, pipe.scheduler.config.num_train_timesteps, (bsz,), device=accelerator.device).long()
            noise = torch.randn_like(latents, device=accelerator.device, dtype=latents.dtype)
            noisy = pipe.scheduler.add_noise(latents, noise, timesteps)

            encoder_hidden_states = pipe.encode_prompt(
                captions,
                accelerator.device,
                1,
                do_classifier_free_guidance=False,
            )
            noise_pred = unet(noisy, timesteps, encoder_hidden_states).sample
            loss = torch.nn.functional.mse_loss(noise_pred, noise, reduction="mean")
            accelerator.backward(loss)
            if accelerator.sync_gradients:
                accelerator.clip_grad_norm_(unet.parameters(), 1.0)
            optimizer.step()
            lr_scheduler.step()
            optimizer.zero_grad()

        global_step += 1

    accelerator.wait_for_everyone()
    if accelerator.is_main_process:
        unwrapped = accelerator.unwrap_model(unet)
        unwrapped.save_pretrained(str(output_path))
        (output_path / "metadata.json").write_text(
            json.dumps({"steps": global_step, "rank": args.rank}, indent=2),
            encoding="utf-8",
        )


if __name__ == "__main__":
    main()
