#!/usr/bin/env python3
"""
워크플로우 자동 패치: ltx23_i2v.json 에 VHS_LoadVideo 노드를 추가하고
pipelines/ltx23_i2v_ref.json 으로 저장. (레퍼런스 댄스 영상 주입 시 자동 사용됨)

실행: backend 디렉터리에서
  python scripts/patch_workflow.py
또는
  python -m scripts.patch_workflow
"""
from __future__ import annotations

import json
import shutil
import sys
from pathlib import Path

# backend 디렉터리 기준 경로
BACKEND_DIR = Path(__file__).resolve().parent.parent
PIPELINES_DIR = BACKEND_DIR / "pipelines"
CANDIDATES = [
    BACKEND_DIR / "pipelines" / "ltx23_i2v.json",
    BACKEND_DIR / "ltx23_i2v.json",
    BACKEND_DIR / "workflows" / "ltx23_i2v.json",
]
OUTPUT_PATH = PIPELINES_DIR / "ltx23_i2v_ref.json"

LOAD_VIDEO_TYPES = ("VHS_LoadVideo", "VHS_LoadVideoPath", "LoadVideo")


def find_workflow() -> Path:
    for p in CANDIDATES:
        if p.exists():
            return p
    raise FileNotFoundError(
        f"ltx23_i2v.json 을 찾을 수 없습니다. 탐색 경로: {[str(c) for c in CANDIDATES]}"
    )


def get_graph(raw: dict) -> tuple[dict, bool]:
    """(노드 딕셔너리, prompt 키로 감싸져 있었는지) 반환."""
    if isinstance(raw.get("prompt"), dict):
        return raw["prompt"], True
    return raw, False


def get_wrapped_output(raw: dict, graph: dict, use_prompt_key: bool) -> dict:
    """저장용 최상위 구조 반환."""
    if use_prompt_key:
        return {"prompt": graph, **{k: v for k, v in raw.items() if k != "prompt"}}
    return graph


def next_node_id(graph: dict) -> str:
    """기존 노드 ID와 겹치지 않는 새 ID. 숫자만 있는 ID 중 최대+1, 없으면 300."""
    numeric = []
    for nid in graph:
        try:
            numeric.append(int(nid))
        except ValueError:
            pass
    return str(max(numeric, default=299) + 1)


def patch_workflow() -> None:
    src_path = find_workflow()
    print(f"원본 워크플로우: {src_path}")

    with open(src_path, encoding="utf-8") as f:
        raw = json.load(f)

    graph, use_prompt_key = get_graph(raw)

    # 이미 VHS_LoadVideo 계열 노드가 있으면 복사만
    for nid, node in graph.items():
        if isinstance(node, dict) and node.get("class_type") in LOAD_VIDEO_TYPES:
            print(f"이미 비디오 로더 노드 존재 ({nid}). ltx23_i2v_ref.json 으로 복사만 진행합니다.")
            PIPELINES_DIR.mkdir(parents=True, exist_ok=True)
            out_data = get_wrapped_output(raw, graph, use_prompt_key)
            with open(OUTPUT_PATH, "w", encoding="utf-8") as f:
                json.dump(out_data, f, ensure_ascii=False, indent=2)
            print(f"저장 완료: {OUTPUT_PATH}")
            return

    # VHS_LoadVideo 노드 추가 (기존 연결은 변경하지 않음 — 런타임에 파일명만 주입)
    new_id = next_node_id(graph)
    # _meta.title 은 영문만 사용 (ComfyUI가 노드 타입 조회 시 한글 제목으로 인한 오류 방지)
    vhs_node = {
        "class_type": "VHS_LoadVideo",
        "inputs": {
            "video": "__DANCE_VIDEO_PLACEHOLDER__",
            "force_rate": 8,
            "force_size": "Custom Width x Custom Height",
            "custom_width": 768,
            "custom_height": 512,
            "frame_load_cap": 49,
            "skip_first_frames": 0,
            "select_every_nth": 1,
        },
        "_meta": {"title": "VHS Load Video"},
    }
    graph[new_id] = vhs_node
    print(f"VHS_LoadVideo 노드 추가: ID={new_id} (런타임에 video 파일명 주입)")

    PIPELINES_DIR.mkdir(parents=True, exist_ok=True)
    out_data = get_wrapped_output(raw, graph, use_prompt_key)
    with open(OUTPUT_PATH, "w", encoding="utf-8") as f:
        json.dump(out_data, f, ensure_ascii=False, indent=2)
    print(f"패치 완료: {OUTPUT_PATH}")
    print("레퍼런스 영상 사용 시 백엔드가 이 파일을 자동 사용합니다. 서버 재시작 후 적용됩니다.")


if __name__ == "__main__":
    try:
        patch_workflow()
    except FileNotFoundError as e:
        print(e, file=sys.stderr)
        sys.exit(1)
    except Exception as e:
        print(f"패치 실패: {e}", file=sys.stderr)
        sys.exit(1)
