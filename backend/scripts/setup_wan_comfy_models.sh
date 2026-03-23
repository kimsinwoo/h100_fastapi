#!/usr/bin/env bash
# Thin wrapper — see docs/COMFYUI_WAN_MODELS.md
set -euo pipefail
ROOT="$(cd "$(dirname "$0")" && pwd)"
exec python3 "$ROOT/setup_wan_comfy_models.py" "$@"
