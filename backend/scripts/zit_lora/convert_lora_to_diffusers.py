"""
기존 PEFT 전체 저장(old) 형식 .safetensors 를 diffusers 로드 가능 형식으로 변환.
재학습 없이 한 번만 실행하면 됨.

Usage:
  python convert_lora_to_diffusers.py backend/lora_output/cyberpunk.safetensors
  python convert_lora_to_diffusers.py backend/lora_output/cyberpunk.safetensors -o backend/lora_output/cyberpunk_diffusers.safetensors
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

PEFT_PREFIX = "base_model.model."


def main() -> int:
    ap = argparse.ArgumentParser(description="Convert old PEFT LoRA safetensors to diffusers format.")
    ap.add_argument("input", type=Path, help="Input .safetensors file (old format)")
    ap.add_argument("-o", "--output", type=Path, default=None, help="Output path (default: overwrite input)")
    args = ap.parse_args()
    inp = args.input.resolve()
    if not inp.exists():
        print(f"Error: not found: {inp}", file=sys.stderr)
        return 1
    out = args.output.resolve() if args.output else inp

    import safetensors.torch as st
    with st.safe_open(inp, framework="pt", device="cpu") as f:
        state_dict = {k: f.get_tensor(k) for k in f.keys()}

    if not any("base_model.model." in k for k in state_dict):
        print("File is not old PEFT format (no base_model.model.* keys). Nothing to convert.", file=sys.stderr)
        return 0

    converted = {}
    for k, v in state_dict.items():
        if not k.startswith(PEFT_PREFIX):
            continue
        if "lora_A" not in k and "lora_B" not in k:
            continue
        new_key = k.replace(PEFT_PREFIX, "", 1)
        converted[new_key] = v

    if not converted:
        print("No lora_A/lora_B keys found in file. Cannot convert. Re-train with current train_lora_zit.py.", file=sys.stderr)
        print("Sample keys:", list(state_dict.keys())[:10], file=sys.stderr)
        return 1

    st.save_file(converted, out)
    print(f"Saved diffusers-format LoRA: {out} ({len(converted)} keys)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
