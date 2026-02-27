#!/usr/bin/env bash
# Example: invoke POST /edit with one image and prompt.
# Usage: ./scripts/curl_edit.sh [path/to/image.png] [prompt]
set -e
IMAGE_PATH="${1:-./test_image.png}"
PROMPT="${2:-Make the background blue}"
BASE_URL="${BASE_URL:-http://localhost:8000}"

if [ ! -f "$IMAGE_PATH" ]; then
  echo "Create or set IMAGE_PATH to an existing image. Example:"
  echo "  ./scripts/curl_edit.sh /path/to/image.png \"Add a red hat\""
  exit 1
fi

echo "POST $BASE_URL/edit image=$IMAGE_PATH prompt=\"$PROMPT\""
curl -s -X POST "$BASE_URL/edit" \
  -F "image=@$IMAGE_PATH" \
  -F "prompt=$PROMPT" \
  -F "seed=42" \
  -F "steps=28" \
  --output result_edit.png

if [ -f result_edit.png ] && [ -s result_edit.png ]; then
  echo "Saved result to result_edit.png"
else
  echo "Request failed or empty response. Check server logs or response body."
  head -c 500 result_edit.png 2>/dev/null || true
fi
