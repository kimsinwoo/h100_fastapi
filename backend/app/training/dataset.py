"""
Dreambooth-style dataset: dataset/images/ + captions.txt (filename\tcaption per line).
"""

from __future__ import annotations

from pathlib import Path


def load_captions(dataset_path: str | Path) -> list[tuple[str, str]]:
    path = Path(dataset_path)
    images_dir = path / "images"
    captions_file = path / "captions.txt"
    if not captions_file.exists():
        raise FileNotFoundError(f"captions.txt not found in {path}")
    pairs: list[tuple[str, str]] = []
    for line in captions_file.read_text(encoding="utf-8", errors="replace").strip().splitlines():
        line = line.strip()
        if not line:
            continue
        if "\t" in line:
            filename, caption = line.split("\t", 1)
        else:
            parts = line.split(maxsplit=1)
            filename = parts[0]
            caption = parts[1] if len(parts) > 1 else ""
        filename = filename.strip()
        caption = caption.strip()
        img_path = images_dir / filename
        if img_path.exists():
            pairs.append((str(img_path), caption))
    return pairs
