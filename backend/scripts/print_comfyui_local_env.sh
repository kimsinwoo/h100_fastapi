#!/usr/bin/env bash
# talktailForPet 레포: ComfyUI 가 레포 루트 ./ComfyUI 에 있을 때 .env 에 넣을 줄 출력
# 사용: cd zimage_webapp/backend && bash scripts/print_comfyui_local_env.sh

set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
# backend/scripts -> backend -> zimage_webapp -> talktailForPet
ROOT="$(cd "$SCRIPT_DIR/../../.." && pwd)"
COMFY="$ROOT/ComfyUI"

echo "# --- ComfyUI (레포 루트: $ROOT) ---"
echo "# 아래를 zimage_webapp/backend/.env 에 복사 (경로 확인 후 사용)"
echo "COMFYUI_ENABLED=true"
echo "COMFYUI_BASE_URL=http://127.0.0.1:8188"
echo "COMFYUI_REFERENCE_VIDEO_DIR=$COMFY/input"
echo "COMFYUI_OUTPUT_DIR=$COMFY/output"
echo ""
if [[ ! -d "$COMFY" ]]; then
  echo "# WARN: $COMFY 디렉터리가 없습니다. ComfyUI 클론 경로가 다르면 위 두 경로를 수동으로 수정하세요." >&2
fi
