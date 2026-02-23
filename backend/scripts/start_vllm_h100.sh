#!/usr/bin/env bash
# Start vLLM OpenAI-compatible server optimized for single H100.
# Run from backend dir. Requires: pip install vllm (in venv recommended).
#
# 중요: vLLM 기동 전 GPU를 다른 프로세스가 쓰고 있으면 CUDA OOM으로 실패합니다.
#   nvidia-smi  로 확인 후, 다른 GPU 프로세스를 종료한 뒤 실행하세요.
# OOM이 나면: VLLM_ENFORCE_EAGER=1 ./scripts/start_vllm_h100.sh (CUDA 그래프 비활성화, 느리지만 메모리 절약)

set -e
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
BACKEND_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"

# Prefer backend venv so we use the same env where vllm is installed
if [ -x "$BACKEND_DIR/venv/bin/python" ]; then
  PYTHON="$BACKEND_DIR/venv/bin/python"
  export PATH="$BACKEND_DIR/venv/bin:$PATH"
else
  PYTHON="${PYTHON:-python3}"
fi

MODEL="${VLLM_MODEL:-openai/gpt-oss-20b}"
PORT="${VLLM_PORT:-7001}"

# OOM 방지: CUDA 그래프 캡처 시 메모리 여유 필요. 다른 프로세스가 GPU 사용 중이면 반드시 종료 후 실행.
# VLLM_ENFORCE_EAGER=1 이면 CUDA 그래프 비활성화(메모리 절약, 추론은 조금 느림)
ENFORCE_EAGER="${VLLM_ENFORCE_EAGER:-0}"
GPU_UTIL="${VLLM_GPU_MEMORY_UTILIZATION:-0.85}"
MAX_NUM_SEQS="${VLLM_MAX_NUM_SEQS:-64}"
MAX_MODEL_LEN="${VLLM_MAX_MODEL_LEN:-32768}"

# Check vLLM is importable
if ! "$PYTHON" -c "import vllm" 2>/dev/null; then
  echo "Error: vLLM is not installed. Install it in your environment, e.g.:" >&2
  echo "  source $BACKEND_DIR/venv/bin/activate" >&2
  echo "  pip install vllm" >&2
  exit 1
fi

# 공통 인자 (메모리 절감: gpu-memory-utilization 0.85, max-num-seqs 64, max-model-len 32768)
VLLM_EXTRA=(
  --port "$PORT"
  --host 0.0.0.0
  --gpu-memory-utilization "$GPU_UTIL"
  --max-num-seqs "$MAX_NUM_SEQS"
  --max-model-len "$MAX_MODEL_LEN"
  --tensor-parallel-size 1
  --dtype bfloat16
  --trust-remote-code
)
[ "$ENFORCE_EAGER" = "1" ] && VLLM_EXTRA+=( --enforce-eager )

# Prefer `vllm serve` (v0.4+); fallback to module
if command -v vllm &>/dev/null; then
  exec vllm serve "$MODEL" "${VLLM_EXTRA[@]}" "$@"
else
  exec "$PYTHON" -m vllm.entrypoints.openai.api_server \
    --model "$MODEL" \
    "${VLLM_EXTRA[@]}" \
    "$@"
fi

# Optional: --hf-token "$HF_TOKEN" (gated models)
