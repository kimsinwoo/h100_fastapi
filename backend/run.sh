#!/usr/bin/env bash
# 설정 없이 바로 실행: 라이브러리 설치 후 서버 시작
set -e
cd "$(dirname "$0")"

echo "Installing dependencies..."
pip install -q -r requirements.txt

echo "Starting Z-Image AI server on http://0.0.0.0:7000"
exec uvicorn app.main:app --host 0.0.0.0 --port 7000
