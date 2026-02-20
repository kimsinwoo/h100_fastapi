#!/usr/bin/env python3
"""
리밸런스 + 경고문 없음 + 교정 데이터를 합쳐 재학습용 최종 JSONL 생성.
순서: rebalanced, no_warning, correction 합친 뒤 셔플.
"""
from __future__ import annotations

import json
import random
import sys
from pathlib import Path

SCRIPT_DIR = Path(__file__).resolve().parent


def main() -> None:
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--rebalanced", default=str(SCRIPT_DIR / "korean_sft_rebalanced.jsonl"))
    ap.add_argument("--no-warning", default=str(SCRIPT_DIR / "korean_sft_no_warning.jsonl"))
    ap.add_argument("--correction", default=str(SCRIPT_DIR / "korean_sft_correction.jsonl"))
    ap.add_argument("-o", "--output", default=str(SCRIPT_DIR / "korean_sft_final.jsonl"))
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()
    random.seed(args.seed)
    items: list[dict] = []
    for name, path in [
        ("rebalanced", args.rebalanced),
        ("no_warning", args.no_warning),
        ("correction", args.correction),
    ]:
        p = Path(path)
        if not p.exists():
            print(f"건너뜀 (없음): {p}", file=sys.stderr)
            continue
        with open(p, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    items.append(json.loads(line))
                except json.JSONDecodeError:
                    continue
        print(f"로드 {name}: {p}", file=sys.stderr)
    random.shuffle(items)
    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "w", encoding="utf-8") as f:
        for item in items:
            f.write(json.dumps(item, ensure_ascii=False) + "\n")
    print(f"총 {len(items)}건 → {out}", file=sys.stderr)


if __name__ == "__main__":
    main()
