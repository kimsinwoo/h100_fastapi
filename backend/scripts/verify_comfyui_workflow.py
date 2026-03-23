#!/usr/bin/env python3
"""
Lightweight structural checks for ComfyUI API-format workflow JSON.

Does NOT validate that class_type names exist in your ComfyUI install.

Usage:
  python scripts/verify_comfyui_workflow.py pipelines/ltx23_i2v.json --mode ltx
  python scripts/verify_comfyui_workflow.py pipelines/dance/dog_pose_generation.json --mode pose
  python scripts/verify_comfyui_workflow.py pipelines/dance/my_export.json --mode pose --strict
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any


def unwrap_nodes(workflow: dict[str, Any]) -> dict[str, Any]:
    if "prompt" in workflow and isinstance(workflow["prompt"], dict):
        return workflow["prompt"]
    return workflow


def iter_nodes(workflow: dict[str, Any]) -> list[tuple[str, dict[str, Any]]]:
    nodes = unwrap_nodes(workflow)
    out: list[tuple[str, dict[str, Any]]] = []
    for nid, node in nodes.items():
        if isinstance(node, dict) and "class_type" in node:
            out.append((str(nid), node))
    return out


def check_ltx(workflow: dict[str, Any]) -> list[str]:
    errors: list[str] = []
    nodes = iter_nodes(workflow)
    load_image = [
        (nid, n)
        for nid, n in nodes
        if "loadimage" in (n.get("class_type") or "").lower().replace(" ", "")
    ]
    if not load_image:
        errors.append("No LoadImage-like node found (need at least one for i2v).")

    clip_like = [
        (nid, n)
        for nid, n in nodes
        if "clip" in (n.get("class_type") or "").lower()
        and "text" in (n.get("class_type") or "").lower()
    ]
    if len(clip_like) < 1:
        errors.append("No CLIPTextEncode-like nodes found (positive/negative injection expects ≥1).")

    video_loaders = []
    for nid, n in nodes:
        ct = (n.get("class_type") or "").lower().replace(" ", "")
        if ("load" in ct and "video" in ct) or ct in ("vhs_loadvideo", "vhs_loadvideopath", "loadvideo"):
            video_loaders.append((nid, n))

    if video_loaders:
        nid, n = video_loaders[0]
        inputs = n.get("inputs") or {}
        if not any(k in inputs for k in ("video", "file_path", "path", "filename", "input")):
            errors.append(
                f"Video loader node {nid} has no injectable key "
                "(video/file_path/path/filename/input). Backend may add 'video' at runtime."
            )
    return errors


def check_pose(workflow: dict[str, Any], *, strict: bool) -> tuple[list[str], list[str]]:
    """Returns (errors, warnings)."""
    errors: list[str] = []
    warnings: list[str] = []
    nodes = iter_nodes(workflow)
    load_image = [
        (nid, n)
        for nid, n in nodes
        if "loadimage" in (n.get("class_type") or "").lower().replace(" ", "")
    ]
    if len(load_image) < 2:
        errors.append(
            f"Expected ≥2 LoadImage nodes (character + pose), found {len(load_image)}."
        )

    clip_like = [
        (nid, n)
        for nid, n in nodes
        if "clip" in (n.get("class_type") or "").lower()
        and "text" in (n.get("class_type") or "").lower()
    ]
    samplers = [
        (nid, n)
        for nid, n in nodes
        if "sampler" in (n.get("class_type") or "").lower()
    ]

    if strict:
        if len(clip_like) < 2:
            errors.append(
                f"[strict] Expected ≥2 CLIPTextEncode-like nodes for pos/neg, found {len(clip_like)}."
            )
        if not samplers:
            errors.append("[strict] No KSampler / KSamplerAdvanced-like node found.")
        else:
            has_seed = False
            for _nid, n in samplers:
                inp = n.get("inputs") or {}
                if isinstance(inp, dict) and "seed" in inp:
                    has_seed = True
                    break
            if not has_seed:
                errors.append(
                    "[strict] Sampler nodes found but none have inputs.seed "
                    "(backend only patches existing seed)."
                )
    else:
        if len(clip_like) < 2:
            warnings.append(
                f"Only {len(clip_like)} CLIPTextEncode-like node(s). "
                "Production pose_sdxl export should have ≥2 (pos/neg)."
            )
        if not samplers:
            warnings.append(
                "No KSampler-like node — shipped minimal template? Replace with full ComfyUI export for pose_sdxl."
            )
        elif not any(
            isinstance(n.get("inputs"), dict) and "seed" in (n.get("inputs") or {})
            for _nid, n in samplers
        ):
            warnings.append(
                "Sampler has no inputs.seed — backend will not inject seed until you add it in the graph."
            )

    return errors, warnings


def main() -> int:
    p = argparse.ArgumentParser(description="Verify ComfyUI workflow JSON structure.")
    p.add_argument("json_path", type=Path, help="Path to workflow JSON")
    p.add_argument("--mode", choices=("ltx", "pose"), required=True)
    p.add_argument(
        "--strict",
        action="store_true",
        help="For pose mode: require CLIP×2, KSampler with seed (production export).",
    )
    args = p.parse_args()

    path = args.json_path
    if not path.is_file():
        print(f"ERROR: file not found: {path}", file=sys.stderr)
        return 2

    with path.open(encoding="utf-8") as f:
        data = json.load(f)

    if not isinstance(data, dict):
        print("ERROR: root must be a JSON object", file=sys.stderr)
        return 2

    if args.mode == "ltx":
        errs = check_ltx(data)
        warns: list[str] = []
    else:
        errs, warns = check_pose(data, strict=args.strict)

    print(f"File: {path}")
    print(f"Mode: {args.mode}" + (" (strict)" if args.strict else ""))
    for w in warns:
        print(f"WARN: {w}")
    if errs:
        print("FAIL:")
        for e in errs:
            print(f"  - {e}")
        return 1

    print("OK (structural checks passed).")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
