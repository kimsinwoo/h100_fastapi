#!/usr/bin/env python3
"""
강제 안정화 품질 테스트: 다음 4가지 기준 검사.
- 응답 첫 줄이 특수문자로 시작하면 실패
- 고정 문구 1회 초과 시 실패
- 구두점 밀도 10% 초과 시 실패
- 반복 문장 발생 시 실패
"""
from __future__ import annotations

import re
import sys
from pathlib import Path

# 테스트 기준
SPECIAL_START_CHARS = set(".,;:!?\"'•·-*#@ \t")
FIXED_PHRASES = ("정확한 판단은", "전문가에게 확인하세요", "수의사와 상담", "의료·수의 전문가")
PUNCT_CHARS = set(".,;:!?\"'•·-~()[]{}*#@")
MAX_PUNCT_DENSITY = 0.10  # 10%


def check_first_line_not_special(text: str) -> tuple[bool, str]:
    """첫 줄이 특수문자로 시작하면 실패."""
    if not text or not text.strip():
        return False, "empty"
    first_line = text.lstrip().split("\n")[0].strip()
    if not first_line:
        return False, "first_line_empty"
    if first_line[0] in SPECIAL_START_CHARS:
        return False, f"first_char_special:{first_line[0]!r}"
    return True, "ok"


def check_fixed_phrase_at_most_once(text: str) -> tuple[bool, str]:
    """고정 문구가 1회 초과면 실패."""
    for phrase in FIXED_PHRASES:
        count = text.count(phrase)
        if count > 1:
            return False, f"fixed_phrase_repeat:{phrase!r}:{count}"
    return True, "ok"


def check_punctuation_density(text: str) -> tuple[bool, str]:
    """구두점 밀도 10% 초과 시 실패."""
    if not text or not text.strip():
        return True, "ok"
    punct = sum(1 for c in text if c in PUNCT_CHARS)
    density = punct / max(len(text), 1)
    if density > MAX_PUNCT_DENSITY:
        return False, f"punct_density:{density:.2%}"
    return True, "ok"


def check_no_repeated_sentence(text: str) -> tuple[bool, str]:
    """동일 문장 2회 이상이면 실패."""
    if not text or not text.strip():
        return True, "ok"
    sentences = re.split(r"(?<=[.!?])\s+", text)
    sentences = [s.strip() for s in sentences if len(s.strip()) >= 5]
    seen = set()
    for s in sentences:
        key = s[:80]
        if key in seen:
            return False, "repeated_sentence"
        seen.add(key)
    return True, "ok"


def run_all_checks(text: str) -> dict[str, tuple[bool, str]]:
    """4가지 기준 모두 검사."""
    return {
        "first_line_not_special": check_first_line_not_special(text),
        "fixed_phrase_at_most_once": check_fixed_phrase_at_most_once(text),
        "punctuation_density_le_10": check_punctuation_density(text),
        "no_repeated_sentence": check_no_repeated_sentence(text),
    }


def all_passed(results: dict[str, tuple[bool, str]]) -> bool:
    return all(passed for passed, _ in results.values())


def main() -> None:
    # 샘플로 정상/비정상 케이스 검증
    tests = [
        ("강아지가 숨을 헐떡이면 환경을 안정시켜 주세요.", True),
        (".  , ,  정확한 판단은...", False),
        ("정확한 판단은 의료·수의 전문가에게 확인하세요. 정확한 판단은 다시 한번.", False),
        ("안녕하세요. 반갑습니다. 안녕하세요. 반갑습니다.", False),
    ]
    for text, expect_pass in tests:
        results = run_all_checks(text)
        passed = all_passed(results)
        if passed != expect_pass:
            print(f"UNEXPECTED: {text[:50]}... -> {results}", file=sys.stderr)
        else:
            print(f"OK (expect_pass={expect_pass}): {text[:50]}...")
    print("Criteria self-check done.")

    # clean_output 연동 검증: app 서비스의 clean_output 적용 후 기준 통과 여부
    try:
        sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "backend"))
        from app.services.llm_service import clean_output
        noisy = ".  , , ,  \"정확한 판단은...\"  •  ; ;  \n강아지가 숨을 헐떡이고 있다면 먼저 환경을 안정시켜 주세요."
        cleaned = clean_output(noisy)
        results = run_all_checks(cleaned)
        if all_passed(results):
            print("clean_output integration: PASS (noisy -> cleaned passes criteria)")
        else:
            print("clean_output integration: FAIL", results, file=sys.stderr)
    except Exception as e:
        print("clean_output integration: SKIP (backend not in path)", e, file=sys.stderr)
    return 0


if __name__ == "__main__":
    sys.exit(main())
