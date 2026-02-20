#!/bin/bash
# H100 서버 전용: 재학습 파이프라인 (Flash Attention·bf16·배치 크기 활용)
# 실행 전: pip install flash-attn --no-build-isolation (선택, 없으면 SDPA 사용)

set -e
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
DATA_DIR="${SCRIPT_DIR}/data"
cd "$SCRIPT_DIR"

echo "[1/6] 데이터 정제 (cleaned 없으면 build 후 정제)"
if [ ! -f "${DATA_DIR}/korean_sft_train_cleaned.jsonl" ]; then
  if [ ! -f "${DATA_DIR}/korean_sft_train.jsonl" ]; then
    python data/build_korean_sft.py
  fi
  python data/clean_dataset.py "${DATA_DIR}/korean_sft_train.jsonl" -o "${DATA_DIR}/korean_sft_train_cleaned.jsonl" --diff 2>/dev/null || true
fi

echo "[2/6] 경고문 비율 5% 이하 리밸런스"
python data/rebalance_warning_ratio.py "${DATA_DIR}/korean_sft_train_cleaned.jsonl" -o "${DATA_DIR}/korean_sft_rebalanced.jsonl" --max-ratio 0.05

echo "[3/6] 경고문 없는 정상 데이터 3000건"
python data/build_no_warning_sft.py -n 3000 -o "${DATA_DIR}/korean_sft_no_warning.jsonl"

echo "[4/6] 노이즈→정상 교정 데이터 3500건"
python data/build_correction_sft.py -n 3500 -o "${DATA_DIR}/korean_sft_correction.jsonl"

echo "[5/6] 재학습용 최종 데이터 병합"
python data/merge_for_relearn.py -o "${DATA_DIR}/korean_sft_final.jsonl"

echo "[6/6] H100: bf16, Flash Attention 시도, 배치 4, 5e-5 1 epoch"
python train_lora.py \
  --data_file "${DATA_DIR}/korean_sft_final.jsonl" \
  --learning_rate 5e-5 \
  --num_epochs 1 \
  --per_device_train_batch_size 4 \
  --gradient_accumulation_steps 2 \
  --output_dir "${SCRIPT_DIR}/output/korean_lora"

echo "H100 재학습 파이프라인 완료."
