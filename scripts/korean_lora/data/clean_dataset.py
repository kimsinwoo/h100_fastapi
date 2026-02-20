#!/usr/bin/env python3
"""
SFT JSONL 데이터셋 정제: 반복 따옴표·구두점·문장·JSON 조각·지시문 제거.
클리닝 전후 diff 로그 출력 옵션 지원.
"""
from __future__ import annotations

import json
import re
import sys
from pathlib import Path


# 연속 구두점 2회 이상 → 1회
REPEAT_PUNCT = re.compile(r"([.,;:!?\-~]\s*)\1+")

# 공백 3회 이상 → 1회
MULTI_SPACE = re.compile(r" {3,}")

# 불완전/연속 따옴표 정리
QUOTE_CLEANUP = re.compile(r'"+|\s*"\s*"\s*|\s*"\s*"\s*"')

# system 지시문 패턴 (한글)
SYSTEM_INSTRUCTION = re.compile(
    r"(당신은|답변은|한글만|알파벳|로마자|쓰지\s*않는다|접두어|소제목|목록은|반드시|정확한\s*판단은\s*의료).{0,100}",
    re.IGNORECASE,
)


def _remove_json_fragments(s: str) -> str:
    """assistant 출력에 섞인 JSON 조각만 제거 (일반 따옴표 문장은 유지)."""
    # {"key": "value"} 형태 블록 제거
    s = re.sub(r'\s*\{\s*"[^"]+"\s*:\s*[^}]*\}\s*', " ", s)
    s = re.sub(r'\s*\[\s*"[^"]*"\s*(,\s*"[^"]*")*\s*\]\s*', " ", s)
    return s


def _remove_repeated_sentences(s: str) -> str:
    """동일 문장이 2회 이상 나오면 첫 출현만 유지 (문장 단위)."""
    # 문장 구분: . ! ? 로 끝나는 단위
    parts = re.split(r"(?<=[.!?])\s+", s)
    seen = set()
    out = []
    for p in parts:
        p = p.strip()
        if not p or len(p) < 10:
            out.append(p)
            continue
        key = p[:120]
        if key in seen:
            continue
        seen.add(key)
        out.append(p)
    return " ".join(out)


def clean_assistant_content(text: str) -> str:
    """assistant content 필드 정제."""
    if not text or not isinstance(text, str):
        return text
    s = text.strip()
    # 연속 구두점 축소
    s = REPEAT_PUNCT.sub(r"\1", s)
    # 공백 3회 이상 → 1회
    s = MULTI_SPACE.sub(" ", s)
    # 불완전/연속 따옴표 제거 (문맥 보존을 위해 " " " → " 로만 축소)
    s = re.sub(r'"\s*"\s*"+\s*', '" ', s)
    s = re.sub(r'\s*"\s*"\s*', " ", s)
    # JSON 조각 제거
    s = _remove_json_fragments(s)
    # 동일 문장 2회 이상 반복 제거
    s = _remove_repeated_sentences(s)
    # system 지시문 유사 문장 제거
    s = SYSTEM_INSTRUCTION.sub("", s)
    # 동일 문장 2회 이상 반복 제거 (줄 단위)
    lines = []
    seen = set()
    for line in s.split("\n"):
        stripped = line.strip()
        key = stripped[:80] if len(stripped) > 80 else stripped
        if key and key in seen:
            continue
        if key:
            seen.add(key)
        lines.append(line)
    s = "\n".join(lines)
    # 중복 라인 제거 (연속 동일 라인)
    lines = s.split("\n")
    out = []
    for i, line in enumerate(lines):
        if i > 0 and line.strip() == lines[i - 1].strip() and line.strip():
            continue
        out.append(line)
    s = "\n".join(out)
    # 잘린 문자열 (끝이 불완전한 따옴표/괄호) 정리
    s = re.sub(r'["\s]+$', "", s)
    s = re.sub(r'[\(\{\[\s]+$', "", s)
    # 빈 줄 3줄 이상 → 2줄
    s = re.sub(r"\n{3,}", "\n\n", s)
    # 비정상 문자 (제어문자 등) 제거
    s = "".join(c for c in s if c == "\n" or c == "\t" or (ord(c) >= 32 and ord(c) != 127))
    return s.strip()


def clean_item(item: dict, log_diff: bool) -> tuple[dict, bool]:
    """한 건 정제. 변경 여부 반환."""
    messages = item.get("messages") or []
    changed = False
    new_messages = []
    for m in messages:
        role = (m.get("role") or "").strip()
        content = (m.get("content") or "").strip()
        if role == "assistant" and content:
            before = content
            after = clean_assistant_content(content)
            if before != after:
                changed = True
                if log_diff:
                    print("--- [assistant] BEFORE ---", file=sys.stderr)
                    print(before[:500] + ("..." if len(before) > 500 else ""), file=sys.stderr)
                    print("--- [assistant] AFTER ---", file=sys.stderr)
                    print(after[:500] + ("..." if len(after) > 500 else ""), file=sys.stderr)
                    print("---", file=sys.stderr)
            content = after
        new_messages.append({"role": role, "content": content})
    return {"messages": new_messages}, changed


def main() -> None:
    import argparse
    ap = argparse.ArgumentParser(description="SFT JSONL 정제 + diff 로그")
    ap.add_argument("input", nargs="?", default="korean_sft_train.jsonl", help="입력 JSONL")
    ap.add_argument("-o", "--output", default="korean_sft_train_cleaned.jsonl", help="출력 JSONL")
    ap.add_argument("--diff", action="store_true", help="클리닝 전후 diff 로그 출력")
    args = ap.parse_args()
    script_dir = Path(__file__).resolve().parent
    input_path = Path(args.input) if not Path(args.input).is_absolute() else Path(args.input)
    if not input_path.is_absolute():
        input_path = script_dir / input_path
    output_path = Path(args.output) if Path(args.output).is_absolute() else script_dir / args.output
    if not input_path.exists():
        print(f"입력 파일 없음: {input_path}", file=sys.stderr)
        sys.exit(1)
    count = 0
    changed_count = 0
    with open(input_path, "r", encoding="utf-8") as f_in:
        with open(output_path, "w", encoding="utf-8") as f_out:
            for line in f_in:
                line = line.strip()
                if not line:
                    continue
                try:
                    item = json.loads(line)
                except json.JSONDecodeError:
                    continue
                cleaned, changed = clean_item(item, log_diff=args.diff)
                if changed:
                    changed_count += 1
                f_out.write(json.dumps(cleaned, ensure_ascii=False) + "\n")
                count += 1
    print(f"정제 완료: {count}건 (변경 {changed_count}건) → {output_path}", file=sys.stderr)


if __name__ == "__main__":
    main()
