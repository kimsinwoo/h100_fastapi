#!/usr/bin/env bash
# vLLM 서버만 최소 옵션으로 바로 실행. (스크립트 로직 없이 즉시 출력)
# 사용법: cd backend && source venv/bin/activate && bash scripts/run_vllm_minimal.sh
# 또는 아래 한 줄 명령을 그대로 복사해 터미널에서 실행.

set -e
export PYTHONUNBUFFERED=1

MODEL="${VLLM_MODEL:-openai/gpt-oss-20b}"
PORT="${VLLM_PORT:-7001}"

# vllm이 PATH에 있으면 그대로, 없으면 venv 기준
if command -v vllm &>/dev/null; then
  exec vllm serve "$MODEL" \
    --port "$PORT" \
    --host 0.0.0.0 \
    --gpu-memory-utilization 0.85 \
    --max-num-seqs 64 \
    --max-model-len 32768 \
    --dtype bfloat16 \
    --quantization none \
    --trust-remote-code \
    --enforce-eager \
    "$@"
else
  DIR="$(cd "$(dirname "$0")/.." && pwd)"
  exec "$DIR/venv/bin/python" -u -m vllm.entrypoints.openai.api_server \
    --model "$MODEL" \
    --port "$PORT" \
    --host 0.0.0.0 \
    --gpu-memory-utilization 0.85 \
    --max-num-seqs 64 \
    --max-model-len 32768 \
    --dtype bfloat16 \
    --quantization none \
    --trust-remote-code \
    --enforce-eager \
    "$@"
fi
