"""
LoRA 추론 테스트. style 인자로 lora_output/{style}.safetensors 자동 로드.
"""
from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

_SCRIPT_DIR = Path(__file__).resolve().parent
if str(_SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(_SCRIPT_DIR))

from config import LORA_OUTPUT_DIR, STYLE_CONFIGS, ZIT_MODEL_ID

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--style", type=str, required=True, choices=list(STYLE_CONFIGS.keys()), help="LoRA to load: 3d_render, cyberpunk, pixel_art")
    p.add_argument("--prompt", type=str, default=None)
    p.add_argument("--image", type=str, default=None, help="Input image path (img2img)")
    p.add_argument("--output", type=str, default="inference_output.png")
    p.add_argument("--steps", type=int, default=8)
    p.add_argument("--guidance_scale", type=float, default=3.5)
    p.add_argument("--seed", type=int, default=None)
    args = p.parse_args()

    import torch
    from PIL import Image
    from diffusers import ZImageImg2ImgPipeline

    device = "cuda" if torch.cuda.is_available() else "cpu"
    dtype = torch.bfloat16 if (device == "cuda" and torch.cuda.is_bf16_supported()) else torch.float16
    lora_path = LORA_OUTPUT_DIR / f"{args.style}.safetensors"
    if not lora_path.exists():
        logger.error("LoRA not found: %s. Run train_lora_zit.py --style %s first.", lora_path, args.style)
        return 1

    pipe = ZImageImg2ImgPipeline.from_pretrained(
        ZIT_MODEL_ID,
        torch_dtype=dtype,
        variant="fp16" if dtype == torch.float16 else None,
    ).to(device)

    if hasattr(pipe, "load_lora_weights"):
        if lora_path.is_file():
            pipe.load_lora_weights(str(lora_path.parent), weight_name=lora_path.name)
        else:
            pipe.load_lora_weights(str(lora_path))
        logger.info("Loaded LoRA: %s", lora_path)
    else:
        logger.warning("Pipeline has no load_lora_weights; running without LoRA.")

    prompt = args.prompt or f"{args.style.replace('_', ' ')} style"
    if args.image and Path(args.image).exists():
        image = Image.open(args.image).convert("RGB")
    else:
        image = Image.new("RGB", (1024, 1024), (128, 128, 128))
    generator = None
    if args.seed is not None:
        generator = torch.Generator(device=device).manual_seed(args.seed)

    out = pipe(
        prompt=prompt,
        image=image,
        num_inference_steps=args.steps,
        guidance_scale=args.guidance_scale,
        generator=generator,
        output_type="pil",
    )
    img = out.images[0]
    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    img.save(out_path)
    logger.info("Saved: %s", out_path)
    return 0


if __name__ == "__main__":
    sys.exit(main())
