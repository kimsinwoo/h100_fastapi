#!/usr/bin/env bash
# 설정 없이 바로 실행: 라이브러리 설치 후 서버 시작
# 사용법: cd backend && ./run.sh
set -e
cd "$(dirname "$0")"

if [ -d "venv" ]; then
  source venv/bin/activate
fi

# 첫 설치 시 torch/diffusers 등 대용량 패키지로 10~20분 걸릴 수 있음. 타임아웃·바이너리 우선으로 설치
PIP_TIMEOUT="${PIP_INSTALL_TIMEOUT:-900}"
echo "Installing dependencies (timeout=${PIP_TIMEOUT}s, first run may take 10-20 min)..."
if [ -n "${PIP_VERBOSE:-}" ]; then
  pip install -r requirements.txt --timeout "$PIP_TIMEOUT" --prefer-binary
else
  pip install -r requirements.txt --timeout "$PIP_TIMEOUT" --prefer-binary -q
fi

PORT="${PORT:-7000}"
echo "Starting Z-Image AI server on http://0.0.0.0:${PORT}"
exec uvicorn app.main:app --host 0.0.0.0 --port "$PORT"
