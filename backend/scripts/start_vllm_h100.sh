#!/usr/bin/env bash
# Start vLLM OpenAI-compatible server optimized for single H100.
# Run this first; then start the FastAPI gateway (run_vllm_gateway.py).
# Requires: pip install vllm

set -e
MODEL="${VLLM_MODEL:-openai/gpt-oss-20b}"
PORT="${VLLM_PORT:-7001}"

# Prefer `vllm serve` (v0.4+); fallback to module
if command -v vllm &>/dev/null; then
  exec vllm serve "$MODEL" \
    --port "$PORT" \
    --host 0.0.0.0 \
    --gpu-memory-utilization 0.90 \
    --max-num-seqs 256 \
    --tensor-parallel-size 1 \
    --dtype bfloat16 \
    --trust-remote-code \
    "$@"
else
  exec python -m vllm.entrypoints.openai.api_server \
    --model "$MODEL" \
    --port "$PORT" \
    --host 0.0.0.0 \
    --gpu-memory-utilization 0.90 \
    --max-num-seqs 256 \
    --tensor-parallel-size 1 \
    --dtype bfloat16 \
    --trust-remote-code \
    "$@"
fi

# Optional: --enforce-eager (debug); --hf-token "$HF_TOKEN" (gated models)
