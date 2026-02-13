#!/usr/bin/env bash
# 프론트엔드 빌드 후 백엔드 static_frontend 로 복사. AI 서버 하나로 실행 시 사용.
set -e
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
FRONTEND="$ROOT/frontend"
BACKEND="$ROOT/backend"
DEST="$BACKEND/static_frontend"

echo "Building frontend..."
cd "$FRONTEND"
npm ci --omit=optional 2>/dev/null || npm install
npm run build

echo "Copying dist to $DEST"
rm -rf "$DEST"
mkdir -p "$DEST"
cp -r dist/. "$DEST"
echo "Done. Run backend and open http://localhost:8000"
echo "  cd $BACKEND && ./run.sh"
