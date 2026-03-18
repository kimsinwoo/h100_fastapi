#!/usr/bin/env python3
"""워크플로우 JSON 구조 진단: 래핑 여부 및 노드 목록 출력."""
from __future__ import annotations

import json
import sys
from pathlib import Path

BACKEND_DIR = Path(__file__).resolve().parent.parent
PIPELINES = BACKEND_DIR / "pipelines"


def main() -> None:
    for path in sorted(PIPELINES.glob("*.json")):
        print(f"\n{'='*50}")
        print(f"파일: {path.name}")
        try:
            raw = json.loads(path.read_text(encoding="utf-8"))
        except Exception as e:
            print(f"  로드 실패: {e}")
            continue
        if "prompt" in raw and isinstance(raw["prompt"], dict):
            nodes = raw["prompt"]
            print(f"구조: 래핑됨 (prompt 키 존재) → 노드 수: {len(nodes)}")
        else:
            nodes = raw
            print(f"구조: 직접 노드 dict → 노드 수: {len(nodes)}")
        for nid, node in list(nodes.items())[:25]:
            if isinstance(node, dict):
                ct = node.get("class_type", "?")
                inp = list(node.get("inputs", {}).keys())
                print(f"  [{nid}] {ct}")
                print(f"        inputs: {inp}")
        if len(nodes) > 25:
            print(f"  ... 외 {len(nodes) - 25}개 노드")


if __name__ == "__main__":
    main()
    sys.exit(0)
