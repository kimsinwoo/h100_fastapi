#!/usr/bin/env python3
"""
경고문 없는 정상 문장 SFT 데이터 3000건 이상.
assistant 응답에 "정확한 판단은" 등 경고문 없이 자연스러운 한글 문장만.
반려동물 건강 Q&A 스타일, 시작이 한글/숫자인 본문만.
"""
from __future__ import annotations

import json
import random
import sys
from pathlib import Path

SCRIPT_DIR = Path(__file__).resolve().parent
OUTPUT_FILE = SCRIPT_DIR / "korean_sft_no_warning.jsonl"

SYSTEM = "당신은 반려동물 건강 질문 도우미입니다. 참고 정보만 간단히 답합니다. 진단·확정 표현은 하지 않습니다. 응답은 자연스러운 완전한 문장으로 바로 시작하세요."

# 경고문 없는 정상 응답만 (시작이 한글, 15자 이상 권장)
NORMAL_RESPONSES = [
    "강아지가 숨을 헐떡이면 시원한 곳으로 옮기고 물을 조금씩 주세요.",
    "구토가 반복되면 12시간 정도 금식 후 물만 조금씩 주어 보세요.",
    "눈곱이 많으면 눈 주변을 깨끗한 물에 적신 거즈로 살짝 닦아 주시면 됩니다.",
    "설사 시에는 신선한 물을 충분히 주고 식사는 가볍게 해 주세요.",
    "식욕이 없을 때는 스트레스나 환경 변화를 먼저 점검해 보시는 것이 좋습니다.",
    "피부가 빨갛고 긁는다면 알레르기나 벌레 물림 가능성을 생각해 볼 수 있습니다.",
    "기침이 나오면 목에 이물이 걸리지 않았는지 확인해 보시는 것이 좋습니다.",
    "털이 많이 빠진다면 계절성 탈모일 수 있으나 대량 탈모는 병원 검진을 권합니다.",
    "귀를 자주 긁는다면 귀 안을 살펴보시고 냄새나 분비물이 있으면 확인이 필요합니다.",
    "다리를 절거나 짚고 걷는다면 움직임을 줄이고 병원에 보여 주시는 것이 좋습니다.",
    "소변 색이 진하거나 빈도가 달라지면 수분 섭취와 함께 검진을 권합니다.",
    "입에서 냄새가 심해지면 치아·구강 검진을 받아 보시는 것을 추천합니다.",
    "털이 윤기가 없고 푸석할 때는 영양 상태와 스트레스를 점검해 보세요.",
    "갑자기 움직임을 싫어한다면 통증 가능성을 염두에 두고 조심히 관찰해 주세요.",
    "물을 비정상적으로 많이 마시면 당뇨·신장 등 검사가 필요할 수 있습니다.",
    "증상이 지속되면 병원에서 정확히 확인받는 것이 좋습니다.",
    "위 내용은 일반적인 안내이며 반려동물마다 다를 수 있습니다.",
    "급한 호흡이나 실신이 있으면 즉시 병원을 찾아 주세요.",
    "예방 접종과 정기 검진으로 많은 질환을 예방할 수 있습니다.",
    "배가 부르면 움직이기 싫어할 수 있으니 산책은 소화 후에 해 주세요.",
    "코를 골면 비강이 좁거나 비만일 수 있어 목 주변 관리가 도움이 됩니다.",
    "머리를 자꾸 흔들면 귀 염증이나 이물 가능성을 의심해 보시면 됩니다.",
    "침이 많이 나오면 맛있는 냄새나 구강 문제를 생각해 볼 수 있습니다.",
    "활동이 줄면 나이나 관절 통증을 점검해 보시는 것이 좋습니다.",
]

USER_QUESTIONS = [
    "강아지가 숨을 헐떡여요.",
    "구토를 해요.",
    "눈곱이 많아요.",
    "설사를 해요.",
    "밥을 안 먹어요.",
    "피부가 빨개요.",
    "기침을 해요.",
    "털이 많이 빠져요.",
    "귀를 자꾸 긁어요.",
    "다리를 절어요.",
    "소변 색이 진해요.",
    "입에서 냄새가 나요.",
    "털이 푸석해요.",
    "움직이기 싫어해요.",
    "물을 너무 많이 마셔요.",
    "배가 불렀나요?",
    "코를 골아요.",
    "머리를 자꾸 흔들어요.",
    "침이 많이 나와요.",
    "활동이 줄었어요.",
]


def build_one() -> dict:
    user = random.choice(USER_QUESTIONS)
    assistant = random.choice(NORMAL_RESPONSES)
    return {
        "messages": [
            {"role": "system", "content": SYSTEM},
            {"role": "user", "content": user},
            {"role": "assistant", "content": assistant},
        ]
    }


def main() -> None:
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("-n", "--num", type=int, default=3000)
    ap.add_argument("-o", "--output", default=str(OUTPUT_FILE))
    args = ap.parse_args()
    n = max(3000, args.num)
    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        for _ in range(n):
            f.write(json.dumps(build_one(), ensure_ascii=False) + "\n")
    print(f"경고문 없는 정상 SFT {n}건 생성: {out_path}", file=sys.stderr)


if __name__ == "__main__":
    main()
