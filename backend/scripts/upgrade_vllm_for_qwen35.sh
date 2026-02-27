#!/usr/bin/env bash
# Qwen3.5-35B-A3B(qwen3_5_moe) 사용을 위해 vLLM nightly + 최신 transformers 설치.
# 실행 후: export VLLM_MODEL=Qwen/Qwen3.5-35B-A3B && bash scripts/run_vllm_minimal.sh
# 참고: https://huggingface.co/Qwen/Qwen3.5-35B-A3B

set -e
cd "$(dirname "$0")/.."

echo ">>> Transformers 최신(소스) 설치 중..."
pip install --upgrade "git+https://github.com/huggingface/transformers.git"

echo ">>> vLLM nightly 설치 중 (qwen3_5_moe 지원)..."
pip install vllm --pre --upgrade --extra-index-url https://wheels.vllm.ai/nightly

echo ">>> Qwen3.5 'RMSNormGated activation' 오류 패치 적용 (한 번만 실행)"
python scripts/patch_vllm_rmsnorm_gated.py || true

echo ">>> 완료. 다음으로 7001에서 Qwen3.5 실행:"
echo "    export VLLM_MODEL=Qwen/Qwen3.5-35B-A3B"
echo "    export VLLM_PORT=7001"
echo "    bash scripts/run_vllm_minimal.sh"
