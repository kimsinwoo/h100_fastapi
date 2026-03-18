#!/usr/bin/env python3
"""
ltx23_i2v.json 에 VHS_LoadVideo → Resize → LTXVPreprocess → LTXVImgToVideoInplace 체인을 추가하여
댄스 영상이 LTXVImgToVideoInplace(267:249)의 latent 입력으로 연결되도록 패치한다.
실행: backend 디렉터리에서 python scripts/patch_workflow.py
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

BACKEND_DIR = Path(__file__).resolve().parent.parent
PIPELINES_DIR = BACKEND_DIR / "pipelines"
SRC = PIPELINES_DIR / "ltx23_i2v.json"
DST = PIPELINES_DIR / "ltx23_i2v_ref.json"

ID_VHS = "300"
ID_RESIZE = "301"
ID_PREPROC = "302"
ID_DANCE_ENCODE = "303"


def load(path: Path) -> tuple[dict, dict]:
    """(원본 전체, 노드 dict) 반환."""
    with open(path, encoding="utf-8") as f:
        raw = json.load(f)
    if "prompt" in raw and isinstance(raw["prompt"], dict):
        return raw, raw["prompt"]
    return {"prompt": raw}, raw


def save(raw: dict, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(raw, f, ensure_ascii=False, indent=2)
    print(f"저장: {path}")


def patch() -> None:
    if not SRC.exists():
        print(f"원본 없음: {SRC}", file=sys.stderr)
        sys.exit(1)

    raw, nodes = load(SRC)

    for nid, node in nodes.items():
        if isinstance(node, dict) and "VHS_LoadVideo" in (node.get("class_type") or ""):
            print(f"이미 VHS_LoadVideo 존재 (노드 {nid}). 저장만 진행.")
            save(raw, DST)
            return

    # VHS_LoadVideo
    nodes[ID_VHS] = {
        "class_type": "VHS_LoadVideo",
        "inputs": {
            "video": "__DANCE_VIDEO_PLACEHOLDER__",
            "force_rate": 24,
            "force_size": "Custom Width x Custom Height",
            "custom_width": 720,
            "custom_height": 1280,
            "frame_load_cap": 121,
            "skip_first_frames": 0,
            "select_every_nth": 1,
            "meta_batch_size": 16,
        },
        "_meta": {"title": "VHS Load Video"},
    }
    print(f"노드 추가: [{ID_VHS}] VHS_LoadVideo")

    # ResizeImageMaskNode (댄스 영상 리사이즈) — 267:257(width), 267:258(height) 재사용
    nodes[ID_RESIZE] = {
        "class_type": "ResizeImageMaskNode",
        "inputs": {
            "resize_type": "scale dimensions",
            "resize_type.width": ["267:257", 0],
            "resize_type.height": ["267:258", 0],
            "resize_type.crop": "center",
            "scale_method": "lanczos",
            "input": [ID_VHS, 0],
        },
        "_meta": {"title": "Dance video resize"},
    }
    print(f"노드 추가: [{ID_RESIZE}] ResizeImageMaskNode (댄스용)")

    # LTXVPreprocess (댄스 영상 전처리)
    nodes[ID_PREPROC] = {
        "class_type": "LTXVPreprocess",
        "inputs": {
            "img_compression": 18,
            "image": [ID_RESIZE, 0],
        },
        "_meta": {"title": "Dance preprocess"},
    }
    print(f"노드 추가: [{ID_PREPROC}] LTXVPreprocess (댄스용)")

    # LTXVImgToVideoInplace: 댄스 모션 인코딩 → latent 출력
    nodes[ID_DANCE_ENCODE] = {
        "class_type": "LTXVImgToVideoInplace",
        "inputs": {
            "strength": 0.85,
            "bypass": ["267:201", 0],
            "vae": ["267:236", 2],
            "image": [ID_PREPROC, 0],
            "latent": ["267:228", 0],
        },
        "_meta": {"title": "Dance motion encode"},
    }
    print(f"노드 추가: [{ID_DANCE_ENCODE}] LTXVImgToVideoInplace (댄스 모션)")

    # 267:249 의 latent 를 댄스 인코딩 결과로 교체
    if "267:249" in nodes:
        nodes["267:249"]["inputs"]["latent"] = [ID_DANCE_ENCODE, 0]
        print(f"연결 변경: [267:249].latent ← [{ID_DANCE_ENCODE}] (댄스 모션)")
    else:
        print("노드 267:249 없음. 수동 연결 필요.")

    save(raw, DST)
    print("패치 완료. 레퍼런스 사용 시 백엔드가 ltx23_i2v_ref.json 을 자동 사용합니다.")


if __name__ == "__main__":
    try:
        patch()
    except Exception as e:
        print(f"패치 실패: {e}", file=sys.stderr)
        sys.exit(1)
