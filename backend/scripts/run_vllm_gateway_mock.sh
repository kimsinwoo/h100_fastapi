#!/usr/bin/env bash
# vLLM **게이트웨이**만 Mock 모드로 실행. (실제 vLLM 엔진 없이 테스트용)
# 사용법: cd zimage_webapp/backend && source venv/bin/activate && bash scripts/run_vllm_gateway_mock.sh
# 환경변수: VLLM_USE_MOCK=1, VLLM_PORT=7001 (기본)

set -e
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
BACKEND_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"
cd "$BACKEND_DIR"

export VLLM_USE_MOCK=1
PORT="${VLLM_PORT:-7001}"
export VLLM_PORT="$PORT"

echo ">>> vLLM 게이트웨이 Mock 모드 (포트 $PORT) - vLLM 엔진 불필요"
if [ -x "venv/bin/python" ]; then
  exec venv/bin/python -c "
import uvicorn
from vllm_server.main import app
uvicorn.run(app, host='0.0.0.0', port=$PORT, workers=1, log_level='info')
"
else
  exec python3 -c "
import uvicorn
from vllm_server.main import app
uvicorn.run(app, host='0.0.0.0', port=$PORT, workers=1, log_level='info')
"
fi
