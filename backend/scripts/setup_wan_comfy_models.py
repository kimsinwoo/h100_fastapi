#!/usr/bin/env python3
"""
Download Wan 2.1 / VACE weights into ComfyUI/models/ (flat paths ComfyUI expects).

Uses Hugging Face cache then copies to:
  models/diffusion_models/wan2.1_vace_14B_fp16.safetensors  (Comfy-Org repackaged)
  models/vae/wan_2.1_vae.safetensors
  models/loras/Wan21_*.safetensors  (Kijai)

Usage:
  pip install huggingface_hub
  python scripts/setup_wan_comfy_models.py ~/ComfyUI
  python scripts/setup_wan_comfy_models.py ~/ComfyUI --vae --vace-14b   # large downloads
  python scripts/setup_wan_comfy_models.py ~/ComfyUI --skip-loras

Docs: docs/COMFYUI_WAN_MODELS.md
"""

from __future__ import annotations

import argparse
import errno
import os
import shutil
import sys
from pathlib import Path

# Kijai LoRAs (same filenames as ComfyUI "missing model" dialog)
HF_REPO_KIJAI = "Kijai/WanVideo_comfy"
DEFAULT_LORA_FILES = (
    "Wan21_CausVid_14B_T2V_lora_rank32.safetensors",
    "Wan21_CausVid_bidirect2_T2V_1_3B_lora_rank32.safetensors",
)

# Official Comfy repackaged single files (paths inside repo)
HF_REPO_COMFY_ORG = "Comfy-Org/Wan_2.1_ComfyUI_repackaged"
PATH_VACE_14B = "split_files/diffusion_models/wan2.1_vace_14B_fp16.safetensors"
PATH_VAE = "split_files/vae/wan_2.1_vae.safetensors"


def _is_doc_placeholder(path: Path) -> bool:
    s = str(path.resolve()).replace("\\", "/").lower()
    return "/path/to/" in s


def _hf_get_path(repo_id: str, filename: str) -> Path:
    from huggingface_hub import hf_hub_download  # type: ignore[import-untyped]

    p = hf_hub_download(repo_id=repo_id, filename=filename)
    return Path(p)


def _install_cache_to_dest(cached: Path, dest: Path) -> None:
    """
    Place cached blob at dest without doubling disk use when possible.

    1) os.link (hard link) — same filesystem as ~/.cache/huggingface; no extra bytes.
    2) symlink — if hard link fails (e.g. cross-device).
    3) shutil.copy2 — last resort; needs ~file size free space (Errno 28 if full).
    """
    dest.parent.mkdir(parents=True, exist_ok=True)
    if dest.exists() or dest.is_symlink():
        dest.unlink()

    cached = cached.resolve()
    try:
        os.link(cached, dest)
        mb = dest.stat().st_size // (1024 * 1024)
        print(f"OK (hardlink, no extra disk) -> {dest} ({mb} MB)")
        return
    except OSError:
        pass
    try:
        dest.symlink_to(cached)
        mb = dest.stat().st_size // (1024 * 1024)
        print(f"OK (symlink) -> {dest} ({mb} MB)")
        return
    except OSError:
        pass
    try:
        shutil.copy2(cached, dest)
    except OSError as e:
        if e.errno == errno.ENOSPC:
            print(
                "ERROR: 디스크 공간 부족 (copy 시 원본과 같은 크기가 한 번 더 필요합니다).\n"
                "  HF 캐시와 ComfyUI가 같은 볼륨이면 하드링크가 되어 추가 용량이 거의 안 듭니다.\n"
                "  다른 디스크에 ComfyUI를 두었다면 공간을 비우거나, 외장 SSD 등 여유 있는 경로에 ComfyUI/models 를 두세요.",
                file=sys.stderr,
            )
        raise
    mb = dest.stat().st_size // (1024 * 1024)
    print(f"OK (copy) -> {dest} ({mb} MB)")


def _copy_from_hub(repo_id: str, remote_name: str, dest: Path) -> None:
    """Download to HF cache if needed, then link or copy to dest."""
    cached = _hf_get_path(repo_id, remote_name)
    _install_cache_to_dest(cached, dest)


def ensure_dirs(comfyui: Path) -> None:
    m = comfyui / "models"
    for sub in (
        m / "diffusion_models",
        m / "text_encoders",
        m / "loras",
        m / "vae",
    ):
        sub.mkdir(parents=True, exist_ok=True)


def download_loras(comfyui: Path, filenames: tuple[str, ...]) -> None:
    try:
        from huggingface_hub import hf_hub_download  # type: ignore[import-untyped]
    except ImportError as e:
        print("Missing dependency: pip install huggingface_hub", file=sys.stderr)
        raise SystemExit(1) from e

    dest_dir = comfyui / "models" / "loras"
    dest_dir.mkdir(parents=True, exist_ok=True)
    for name in filenames:
        print(f"Downloading {HF_REPO_KIJAI} / {name} ...")
        # Avoid local_dir= (can nest paths); use cache + flat copy
        cached = Path(hf_hub_download(repo_id=HF_REPO_KIJAI, filename=name))
        out = dest_dir / name
        _install_cache_to_dest(cached, out)
    print("LoRA downloads finished.")


def main() -> int:
    ap = argparse.ArgumentParser(
        description="Wan ComfyUI models: LoRA/VAE/VACE into ComfyUI/models/"
    )
    ap.add_argument("comfyui_root", type=Path, help="ComfyUI installation root")
    ap.add_argument(
        "--skip-loras",
        action="store_true",
        help="Do not download Kijai LoRAs",
    )
    ap.add_argument(
        "--vae",
        action="store_true",
        help=f"Download wan_2.1_vae.safetensors (~242 MB) from {HF_REPO_COMFY_ORG}",
    )
    ap.add_argument(
        "--vace-14b",
        action="store_true",
        help=f"Download wan2.1_vace_14B_fp16.safetensors (~32+ GB) from {HF_REPO_COMFY_ORG}",
    )
    ap.add_argument(
        "-y",
        "--yes",
        action="store_true",
        help="Skip confirmation for very large downloads",
    )
    args = ap.parse_args()
    root = args.comfyui_root.expanduser().resolve()
    if _is_doc_placeholder(root):
        print(
            "ERROR: `/path/to/ComfyUI` 는 문서 예시입니다. 실제 ComfyUI 폴더 경로를 지정하세요.",
            file=sys.stderr,
        )
        return 2
    if root.exists() and not root.is_dir():
        print(f"ERROR: not a directory: {root}", file=sys.stderr)
        return 2
    if not root.exists():
        try:
            root.mkdir(parents=True)
        except OSError as e:
            print(f"ERROR: cannot create {root}: {e}", file=sys.stderr)
            return 2

    ensure_dirs(root)
    models = root / "models"
    print(f"Target: {models}")

    if args.vace_14b and not args.yes:
        print(
            "About to download ~32+ GB (wan2.1_vace_14B_fp16). "
            "Re-run with -y to confirm.",
            file=sys.stderr,
        )
        try:
            confirm = input("Continue? [y/N] ").strip().lower()
        except EOFError:
            confirm = "n"
        if confirm != "y":
            print("Cancelled.")
            return 1

    if not args.skip_loras:
        download_loras(root, DEFAULT_LORA_FILES)
    else:
        print("Skipped LoRA download (--skip-loras).")

    if args.vae:
        print(f"Downloading VAE from {HF_REPO_COMFY_ORG} ...")
        _copy_from_hub(
            HF_REPO_COMFY_ORG,
            PATH_VAE,
            models / "vae" / "wan_2.1_vae.safetensors",
        )

    if args.vace_14b:
        print(f"Downloading VACE 14B from {HF_REPO_COMFY_ORG} (very large) ...")
        _copy_from_hub(
            HF_REPO_COMFY_ORG,
            PATH_VACE_14B,
            models / "diffusion_models" / "wan2.1_vace_14B_fp16.safetensors",
        )

    if not args.skip_loras or args.vae or args.vace_14b:
        print()
    if not (args.vae and args.vace_14b):
        print("Optional next steps:")
        if not args.vae:
            print(f"  VAE (~242 MB): python scripts/setup_wan_comfy_models.py {root} --vae -y")
        if not args.vace_14b:
            print(
                f"  VACE 14B (~32 GB): python scripts/setup_wan_comfy_models.py {root} --vace-14b -y"
            )
    print("Text encoder: place umt5_xxl_fp16 or fp8 under models/text_encoders/ (see docs).")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
