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

# H100 NVL (80/93GB): GPU 단독 사용 시 0.85~0.90 가능. 다른 프로세스(Z-Image 등)와 공유 시 여유 메모리에 맞춰 낮추기.
# "Free memory (72.37/93.0 GiB) is less than desired (0.88, 81.84 GiB)" → VLLM_GPU_MEMORY_UTILIZATION=0.75 로 실행하거나 GPU 선점 프로세스 종료.
# 기본값 0.75: 약 70GiB 요청으로, 72GiB 여유 있을 때도 기동 가능. GPU 단독이면 VLLM_GPU_MEMORY_UTILIZATION=0.88 로 더 높일 수 있음.
ENFORCE_EAGER="${VLLM_ENFORCE_EAGER:-0}"
GPU_UTIL="${VLLM_GPU_MEMORY_UTILIZATION:-0.75}"
MAX_NUM_SEQS="${VLLM_MAX_NUM_SEQS:-96}"
MAX_MODEL_LEN="${VLLM_MAX_MODEL_LEN:-32768}"
# GPU 개수 초과 시 "World size (2) is larger than the number of available GPUs (1)" 방지
NUM_GPUS=1
if command -v nvidia-smi &>/dev/null; then
  NUM_GPUS=$(nvidia-smi -L 2>/dev/null | wc -l)
  [ "$NUM_GPUS" -lt 1 ] && NUM_GPUS=1
fi
REQUESTED_TP="${VLLM_TENSOR_PARALLEL_SIZE:-1}"
if [ "$REQUESTED_TP" -gt "$NUM_GPUS" ]; then
  echo ">>> GPU가 ${NUM_GPUS}대뿐이므로 tensor_parallel_size를 ${REQUESTED_TP} → ${NUM_GPUS}로 조정합니다."
  TENSOR_PARALLEL="$NUM_GPUS"
else
  TENSOR_PARALLEL="$REQUESTED_TP"
fi
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

echo ">>> gpu_memory_utilization: $GPU_UTIL, tensor_parallel_size: $TENSOR_PARALLEL (GPU ${NUM_GPUS}대), 양자화: ${QUANTIZATION:-미전달(모델 mxfp4)}"

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
