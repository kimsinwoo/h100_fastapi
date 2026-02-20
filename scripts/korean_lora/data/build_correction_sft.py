#!/usr/bin/env python3
"""
교정 전용 SFT 데이터 생성: 잘못된 출력(노이즈) → 올바른 출력.
2000~5000개 생성. 초반 노이즈·경고문 강박 제거 학습용.
전체 학습 데이터의 20% 이상 포함할 것.
"""
from __future__ import annotations

import json
import random
import sys
from pathlib import Path

SCRIPT_DIR = Path(__file__).resolve().parent
OUTPUT_FILE = SCRIPT_DIR / "korean_sft_correction.jsonl"

# 노이즈 패턴 (입력에 넣을 잘못된 시작 예시)
NOISE_STARTS = [
    ".  , , ,  '정확한 판단은...'  •  ; ;  ",
    "\"\" \" \"  ,,,  정확한 판단은 의료·수의 전문가에게 확인하세요. 정확한 판단은...  ",
    ".;;  •  •  전문가에게 확인하세요. 수의사와 상담  ",
    ", , . .  \"\"  ;  :  ",
    "• • •  정확한 판단은... 전문가에게 확인하세요.  ",
    "\"  .  ,  \"  ",
    "; : . ,  의료·수의 전문가  수의사와 상담  ",
    ". . .  \"정확한 판단은\"  , ,  ",
]

# 올바른 응답 예시 (반려동물 건강 도우미 스타일)
CORRECT_RESPONSES = [
    "강아지가 숨을 헐떡이고 있다면 먼저 환경을 안정시켜 주세요.",
    "구토가 반복되면 12시간 이상 금식 후 물만 조금씩 주어 보세요.",
    "눈곱이 많다면 눈 주변을 깨끗한 물에 적신 거즈로 닦아 주시면 됩니다.",
    "설사 시에는 신선한 물을 충분히 주고, 식사는 가볍게 해 주세요.",
    "반려동물이 식욕이 없을 때는 스트레스나 환경 변화를 먼저 점검해 보세요.",
    "피부가 빨갛고 긁는다면 알레르기나 벌레 물림 가능성을 생각해 볼 수 있습니다.",
    "기침이 나올 때는 목에 이물이 걸리지 않았는지 확인해 보시는 것이 좋습니다.",
    "털이 많이 빠진다면 계절성 탈모일 수 있으나, 대량 탈모는 병원 검진을 권합니다.",
    "귀를 자주 긁는다면 귀 안을 살펴보시고 냄새나 분비물이 있으면 확인이 필요합니다.",
    "다리를 절거나 짚고 걷는다면 움직임을 줄이고 병원에 보여 주시는 것이 좋습니다.",
    "소변 색이 진하거나 빈도가 달라지면 수분 섭취와 함께 검진을 권합니다.",
    "입에서 냄새가 심해지면 치아·구강 검진을 받아 보시는 것을 추천합니다.",
    "털이 윤기가 없고 푸석할 때는 영양 상태와 스트레스를 점검해 보세요.",
    "갑자기 움직임을 싫어한다면 통증 가능성을 염두에 두고 조심히 관찰해 주세요.",
    "물을 비정상적으로 많이 마시면 당뇨·신장 등 검사가 필요할 수 있습니다.",
]

# 추가 자연 문장 (다양성)
MORE_RESPONSES = [
    "증상이 지속되면 병원에서 정확히 확인받는 것이 좋습니다.",
    "참고로 생각해 볼 수 있는 내용만 말씀드렸습니다. 정확한 판단은 의료·수의 전문가에게 확인하세요.",
    "위 내용은 일반적인 안내이며, 반려동물마다 다를 수 있습니다.",
    "급한 호흡이나 실신이 있으면 즉시 병원을 찾아 주세요.",
    "예방 접종과 정기 검진으로 많은 질환을 예방할 수 있습니다.",
]


def build_one_item() -> dict:
    """잘못된 출력(노이즈) → 올바른 출력 1건."""
    noise = random.choice(NOISE_STARTS)
    correct = random.choice(CORRECT_RESPONSES + MORE_RESPONSES)
    # 사용자 메시지는 "잘못된 출력을 고쳐 주세요" 형태 또는 질문+노이즈
    if random.random() < 0.5:
        user = "위 답변을 자연스러운 문장으로 다시 써 주세요."
        context = noise
    else:
        user = "강아지 건강 질문이에요. 다음처럼 답하지 말고 자연스럽게 한 문장으로만 답해 주세요: " + noise[:80]
        context = ""
    # assistant는 반드시 올바른 한 문장으로 시작
    return {
        "messages": [
            {"role": "system", "content": "응답은 반드시 자연스러운 완전한 문장으로 시작하십시오. 구두점, 따옴표, 특수기호로 시작하지 마십시오. 경고 문구를 자동으로 삽입하지 마십시오."},
            {"role": "user", "content": user if not context else user},
            {"role": "assistant", "content": correct},
        ]
    }


def main() -> None:
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("-n", "--num", type=int, default=3500, help="생성할 샘플 수 (2000~5000)")
    ap.add_argument("-o", "--output", default=str(OUTPUT_FILE))
    args = ap.parse_args()
    n = max(1, min(5000, args.num))
    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    count = 0
    with open(out_path, "w", encoding="utf-8") as f:
        for _ in range(n):
            item = build_one_item()
            f.write(json.dumps(item, ensure_ascii=False) + "\n")
            count += 1
    print(f"교정 SFT 데이터 {count}건 생성: {out_path}", file=sys.stderr)


if __name__ == "__main__":
    main()
