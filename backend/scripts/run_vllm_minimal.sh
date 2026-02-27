#!/usr/bin/env bash
# vLLM 서버 최소 옵션으로 실행 (Linux, H100 NVL 호환). 환경변수로 오버라이드 가능.
# 사용법: cd backend && source venv/bin/activate && bash scripts/run_vllm_minimal.sh
# 참고: 공식 사용법은 pip install vllm && vllm serve "Qwen/Qwen3.5-35B-A3B" (기본 포트 8000).
#       이 스크립트는 메인 앱(7000)과 구분해 포트 7001 사용. 프로덕션: scripts/start_vllm_h100.sh

set -e
export PYTHONUNBUFFERED=1

# 기본값 gpt-oss-20b (vLLM 0.16에서 동작). Qwen3.5: VLLM_MODEL=Qwen/Qwen3.5-35B-A3B
MODEL="${VLLM_MODEL:-openai/gpt-oss-20b}"
PORT="${VLLM_PORT:-7001}"
GPU_UTIL="${VLLM_GPU_MEMORY_UTILIZATION:-0.88}"
MAX_NUM_SEQS="${VLLM_MAX_NUM_SEQS:-96}"
MAX_MODEL_LEN="${VLLM_MAX_MODEL_LEN:-32768}"
# OOM 시: VLLM_GPU_MEMORY_UTILIZATION=0.80 VLLM_MAX_NUM_SEQS=32 또는 VLLM_ENFORCE_EAGER=1
# Qwen3.5 사용: bash scripts/upgrade_vllm_for_qwen35.sh 후 export VLLM_MODEL=Qwen/Qwen3.5-35B-A3B

# VLLM_PORT를 unset해 두면 분산 통신은 자동 포트를 쓰고, API만 --port 로 7001 사용 (API가 7002로 열리는 현상 방지)
unset VLLM_PORT

# Qwen3.5: V1 엔진에서 RMSNormGated 'activation' AttributeError 발생 시 우회 (V0 엔진 사용)
# 사용자가 VLLM_USE_V1=1 로 오버라이드하면 V1 사용 가능(버그 수정된 nightly 이후)
if [[ "$MODEL" == *Qwen3.5* ]]; then
  export VLLM_USE_V1="${VLLM_USE_V1:-0}"
fi

# 지정 포트가 이미 사용 중이면 선점 프로세스 종료 (이전 실행 잔여 시)
if command -v fuser &>/dev/null; then
  fuser -k "${PORT}/tcp" 2>/dev/null || true
  sleep 1
elif command -v ss &>/dev/null; then
  ss -tlnp 2>/dev/null | grep -E ":${PORT}[^0-9]|:${PORT}\s" | sed -n 's/.*pid=\([0-9]*\).*/\1/p' | sort -u | while read -r pid; do
    [ -n "$pid" ] && kill -9 "$pid" 2>/dev/null || true
  done
  sleep 1
fi

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
