#!/usr/bin/env bash
# Run HunyuanImage-3.0-Instruct backend locally (port 8000).
set -e
cd "$(dirname "$0")"
if [ -d "venv" ]; then
  source venv/bin/activate
fi
exec uvicorn app.main:app --host 0.0.0.0 --port 8000 --workers 1 --loop uvloop
