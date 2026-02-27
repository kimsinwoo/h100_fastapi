#!/usr/bin/env bash
# vLLM 서버 최소 옵션으로 실행 (Linux, H100 NVL 호환). 환경변수로 오버라이드 가능.
# 사용법: cd backend && source venv/bin/activate && bash scripts/run_vllm_minimal.sh
# 프로덕션 H100 NVL 권장: scripts/start_vllm_h100.sh (env로 GPU/메모리 조정)

set -e
export PYTHONUNBUFFERED=1

# 기본값 gpt-oss-20b (vLLM 0.16에서 동작). Qwen3.5 쓰려면 scripts/upgrade_vllm_for_qwen35.sh 실행 후 VLLM_MODEL=Qwen/Qwen3.5-35B-A3B
MODEL="${VLLM_MODEL:-openai/gpt-oss-20b}"
PORT="${VLLM_PORT:-7001}"
GPU_UTIL="${VLLM_GPU_MEMORY_UTILIZATION:-0.88}"
MAX_NUM_SEQS="${VLLM_MAX_NUM_SEQS:-96}"
MAX_MODEL_LEN="${VLLM_MAX_MODEL_LEN:-32768}"
# OOM 시: VLLM_GPU_MEMORY_UTILIZATION=0.80 VLLM_MAX_NUM_SEQS=32 또는 VLLM_ENFORCE_EAGER=1
# Qwen3.5 사용: bash scripts/upgrade_vllm_for_qwen35.sh 후 export VLLM_MODEL=Qwen/Qwen3.5-35B-A3B

if command -v vllm &>/dev/null; then
  exec vllm serve "$MODEL" \
    --port "$PORT" \
    --host 0.0.0.0 \
    --gpu-memory-utilization "$GPU_UTIL" \
    --max-num-seqs "$MAX_NUM_SEQS" \
    --max-model-len "$MAX_MODEL_LEN" \
    --dtype bfloat16 \
    --trust-remote-code \
    --enforce-eager \
    "$@"
else
  DIR="$(cd "$(dirname "$0")/.." && pwd)"
  exec "$DIR/venv/bin/python" -u -m vllm.entrypoints.openai.api_server \
    --model "$MODEL" \
    --port "$PORT" \
    --host 0.0.0.0 \
    --gpu-memory-utilization "$GPU_UTIL" \
    --max-num-seqs "$MAX_NUM_SEQS" \
    --max-model-len "$MAX_MODEL_LEN" \
    --dtype bfloat16 \
    --trust-remote-code \
    --enforce-eager \
    "$@"
fi
