#!/usr/bin/env bash
# Start vLLM OpenAI-compatible server optimized for single H100.
# Run from backend dir. Requires: pip install vllm (in venv recommended).
#
# 중요: vLLM 기동 전 GPU를 다른 프로세스가 쓰고 있으면 CUDA OOM으로 실패합니다.
#   nvidia-smi  로 확인 후, 다른 GPU 프로세스를 종료한 뒤 실행하세요.
# OOM이 나면: VLLM_ENFORCE_EAGER=1 ./scripts/start_vllm_h100.sh (CUDA 그래프 비활성화, 느리지만 메모리 절약)
# 기동만 빨리: VLLM_ENFORCE_EAGER=1 (CUDA 그래프 캡처 생략 → 기동 1분 내외, 추론은 약간 느림)
# 양자화: 기본 fp8(H100 고속). fp8 오류 시 VLLM_QUANTIZATION=none ./scripts/start_vllm_h100.sh
# mxfp4 사용하려면 vLLM을 0.15+ 로 업그레이드 후 VLLM_QUANTIZATION= 비우기

set -e
# 실행 직후 콘솔에 바로 출력 (모델 로드 전에 사용자에게 안내)
echo ">>> vLLM 서버 시작 중..."

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

echo ">>> 모델: $MODEL, 포트: $PORT (로드에 1~2분 걸릴 수 있음)"
echo ""

# OOM 방지: CUDA 그래프 캡처 시 메모리 여유 필요. 다른 프로세스가 GPU 사용 중이면 반드시 종료 후 실행.
# VLLM_ENFORCE_EAGER=1 이면 CUDA 그래프 비활성화(메모리 절약, 추론은 조금 느림)
ENFORCE_EAGER="${VLLM_ENFORCE_EAGER:-0}"
GPU_UTIL="${VLLM_GPU_MEMORY_UTILIZATION:-0.85}"
MAX_NUM_SEQS="${VLLM_MAX_NUM_SEQS:-64}"
MAX_MODEL_LEN="${VLLM_MAX_MODEL_LEN:-32768}"
# 양자화: H100에서 fp8=빠른 속도. 모델 기본값 mxfp4는 vLLM 0.15+ 필요. 0.7.x에서는 fp8 또는 none.
# VLLM_QUANTIZATION=fp8 (기본) | none (양자화 끔) | mxfp4 쓰려면 vLLM 업그레이드 후 비우기
QUANTIZATION="${VLLM_QUANTIZATION:-fp8}"

# vllm CLI가 있으면 설치 검사 생략(검사 시 torch/vllm 로드로 10~30초 걸림). 없을 때만 import 검사.
if command -v vllm &>/dev/null; then
  echo ">>> vLLM CLI 사용. 엔진 시작..."
else
  echo ">>> vLLM 설치 확인 중..."
  if ! "$PYTHON" -c "import vllm" 2>/dev/null; then
    echo "Error: vLLM is not installed. Install it in your environment, e.g.:" >&2
    echo "  source $BACKEND_DIR/venv/bin/activate" >&2
    echo "  pip install vllm" >&2
    exit 1
  fi
  echo ">>> vLLM 확인됨. 엔진 시작..."
fi

# 공통 인자 (메모리 절감: gpu-memory-utilization 0.85, max-num-seqs 64, max-model-len 32768)
# 양자화: fp8 = H100에서 고속 추론 (vLLM 0.7.x는 mxfp4 미지원 → fp8 또는 none으로 모델 config 오버라이드)
VLLM_EXTRA=(
  --port "$PORT"
  --host 0.0.0.0
  --gpu-memory-utilization "$GPU_UTIL"
  --max-num-seqs "$MAX_NUM_SEQS"
  --max-model-len "$MAX_MODEL_LEN"
  --tensor-parallel-size 1
  --dtype bfloat16
  --quantization "$QUANTIZATION"
  --trust-remote-code
)
[ "$ENFORCE_EAGER" = "1" ] && VLLM_EXTRA+=( --enforce-eager )

echo ">>> 양자화: $QUANTIZATION (속도 향상 목적)"

# Python/vLLM 로그가 버퍼 없이 바로 콘솔에 나오도록
export PYTHONUNBUFFERED=1

# Prefer `vllm serve` (v0.4+); fallback to module
if command -v vllm &>/dev/null; then
  exec vllm serve "$MODEL" "${VLLM_EXTRA[@]}" "$@"
else
  exec "$PYTHON" -u -m vllm.entrypoints.openai.api_server \
    --model "$MODEL" \
    "${VLLM_EXTRA[@]}" \
    "$@"
fi

# Optional: --hf-token "$HF_TOKEN" (gated models)
