#!/usr/bin/env bash
# Start vLLM OpenAI-compatible server optimized for single H100.
# Run from backend dir. Requires: pip install vllm (in venv recommended).
# Usage: source venv/bin/activate && pip install vllm && ./scripts/start_vllm_h100.sh

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

# Check vLLM is importable
if ! "$PYTHON" -c "import vllm" 2>/dev/null; then
  echo "Error: vLLM is not installed. Install it in your environment, e.g.:" >&2
  echo "  source $BACKEND_DIR/venv/bin/activate" >&2
  echo "  pip install vllm" >&2
  exit 1
fi

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
  exec "$PYTHON" -m vllm.entrypoints.openai.api_server \
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
