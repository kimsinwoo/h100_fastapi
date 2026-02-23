#!/usr/bin/env bash
# Start vLLM OpenAI-compatible server for Linux + NVIDIA H100 NVL.
# Run from backend dir. Requires: pip install vllm (in venv recommended).
#
# 환경: Linux, GPU H100 NVL (1.19.210 등). 단일 GPU 기본; 2-GPU NVL은 VLLM_TENSOR_PARALLEL_SIZE=2.
# 중요: vLLM 기동 전 GPU를 다른 프로세스가 쓰고 있으면 CUDA OOM으로 실패합니다.
#   nvidia-smi  로 확인 후, 다른 GPU 프로세스를 종료한 뒤 실행하세요.
# OOM이 나면: VLLM_ENFORCE_EAGER=1 (CUDA 그래프 비활성화, 메모리 절약)
# 기동만 빨리: VLLM_ENFORCE_EAGER=1 (CUDA 그래프 캡처 생략 → 기동 1분 내외)
# 양자화: 기본 fp8(H100 고속). fp8 오류 시 인자 생략(모델 기본값) 또는 vLLM 0.15+ 업그레이드.

set -e
echo ">>> vLLM 서버 시작 중 (Linux, H100 NVL)..."

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

# H100 NVL (80GB): gpu-memory-utilization 0.85~0.90, max-num-seqs 64~128. 2-GPU NVL이면 tensor_parallel_size=2.
# OOM 시: GPU_UTIL 낮추기(0.80), MAX_NUM_SEQS 줄이기(32), 또는 VLLM_ENFORCE_EAGER=1
ENFORCE_EAGER="${VLLM_ENFORCE_EAGER:-0}"
GPU_UTIL="${VLLM_GPU_MEMORY_UTILIZATION:-0.88}"
MAX_NUM_SEQS="${VLLM_MAX_NUM_SEQS:-96}"
MAX_MODEL_LEN="${VLLM_MAX_MODEL_LEN:-32768}"
TENSOR_PARALLEL="${VLLM_TENSOR_PARALLEL_SIZE:-1}"
# openai/gpt-oss-20b는 모델 config가 mxfp4 → --quantization 전달 시 fp8과 충돌. 비우면 모델 기본값(mxfp4) 사용.
# mxfp4는 vLLM 0.15+ 필요. 0.7.x에서는 "Unknown quantization: mxfp4" 나오면 pip install -U "vllm>=0.15"
QUANTIZATION="${VLLM_QUANTIZATION:-}"

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

# 공통 인자: Linux 0.0.0.0 수신, H100 NVL에 맞춘 bfloat16, 연속 배치(MAX_NUM_SEQS)
VLLM_EXTRA=(
  --port "$PORT"
  --host 0.0.0.0
  --gpu-memory-utilization "$GPU_UTIL"
  --max-num-seqs "$MAX_NUM_SEQS"
  --max-model-len "$MAX_MODEL_LEN"
  --tensor-parallel-size "$TENSOR_PARALLEL"
  --dtype bfloat16
  --trust-remote-code
)
# vLLM 0.7.x는 --quantization none 미지원. none/비어있으면 인자 생략.
[ -n "$QUANTIZATION" ] && [ "$QUANTIZATION" != "none" ] && VLLM_EXTRA+=( --quantization "$QUANTIZATION" )
[ "$ENFORCE_EAGER" = "1" ] && VLLM_EXTRA+=( --enforce-eager )

echo ">>> 양자화: ${QUANTIZATION:-미전달(모델 mxfp4, vLLM 0.15+ 필요)}"

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
