#!/usr/bin/env python3
"""
경고문 비율 5% 이하로 리밸런스.
assistant 출력에 "정확한 판단은" 또는 "의료·수의 전문가" 포함 여부로 분류 후,
경고문 포함 샘플을 전체의 max_ratio 이하로 샘플링.
"""
from __future__ import annotations

import json
import random
import sys
from pathlib import Path

SCRIPT_DIR = Path(__file__).resolve().parent
WARNING_MARKERS = ("정확한 판단은", "의료·수의 전문가", "전문가에게 확인하세요")


def _has_warning(item: dict) -> bool:
    for msg in item.get("messages") or []:
        if (msg.get("role") or "").strip().lower() != "assistant":
            continue
        content = (msg.get("content") or "")
        for m in WARNING_MARKERS:
            if m in content:
                return True
    return False


def main() -> None:
    import argparse
    ap = argparse.ArgumentParser(description="경고문 비율 5%% 이하 리밸런스")
    ap.add_argument("input", nargs="?", default="korean_sft_train_cleaned.jsonl")
    ap.add_argument("-o", "--output", default="korean_sft_rebalanced.jsonl")
    ap.add_argument("--max-ratio", type=float, default=0.05, help="경고문 포함 비율 상한 (0.05 = 5%%)")
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()
    input_path = Path(args.input)
    if not input_path.is_absolute():
        input_path = SCRIPT_DIR / input_path
    output_path = Path(args.output)
    if not output_path.is_absolute():
        output_path = SCRIPT_DIR / output_path
    if not input_path.exists():
        print(f"입력 없음: {input_path}", file=sys.stderr)
        sys.exit(1)

    random.seed(args.seed)
    with_warning: list[dict] = []
    without_warning: list[dict] = []
    with open(input_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                item = json.loads(line)
            except json.JSONDecodeError:
                continue
            if _has_warning(item):
                with_warning.append(item)
            else:
                without_warning.append(item)

    n_no = len(without_warning)
    n_yes = len(with_warning)
    total_after = n_no + min(n_yes, max(0, int((n_no + n_yes) * args.max_ratio / (1 - args.max_ratio)) - n_no))
    # 목표: with_warning / (n_no + with_warning_kept) <= max_ratio
    # with_warning_kept <= (n_no + with_warning_kept) * max_ratio  =>  with_warning_kept <= n_no * max_ratio / (1 - max_ratio)
    max_keep = max(0, int(n_no * args.max_ratio / (1 - args.max_ratio)))
    keep_warning = random.sample(with_warning, min(len(with_warning), max_keep))
    out_items = without_warning + keep_warning
    random.shuffle(out_items)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        for item in out_items:
            f.write(json.dumps(item, ensure_ascii=False) + "\n")

    n_keep = len(keep_warning)
    ratio = n_keep / len(out_items) if out_items else 0
    print(f"리밸런스 완료: 경고문 없음 {n_no}건, 경고문 있음 {n_yes}건 → 유지 {n_keep}건", file=sys.stderr)
    print(f"총 {len(out_items)}건, 경고문 비율 {ratio:.1%} → {output_path}", file=sys.stderr)


if __name__ == "__main__":
    main()
