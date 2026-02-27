#!/usr/bin/env bash
# Run FireRed backend locally. 기존처럼 uvicorn 사용, 포트 7000.
set -e
cd "$(dirname "$0")"
if [ -d "venv" ]; then
  source venv/bin/activate
fi
exec uvicorn app.main:app --host 0.0.0.0 --port 7000 --workers 1 --loop uvloop
