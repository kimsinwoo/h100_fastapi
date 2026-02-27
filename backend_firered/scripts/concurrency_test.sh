#!/usr/bin/env bash
# Send multiple concurrent POST /edit requests. Adjust IMAGE_PATH and BASE_URL as needed.
set -e
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT_DIR="$(cd "$SCRIPT_DIR/../.." && pwd)"
IMAGE_PATH="${IMAGE_PATH:-$ROOT_DIR/../sdxl_dog_editor/outputs/out_1.png}"
BASE_URL="${BASE_URL:-http://localhost:8000}"
CONCURRENCY="${CONCURRENCY:-3}"

if [ ! -f "$IMAGE_PATH" ]; then
  echo "Sample image not found: $IMAGE_PATH"
  echo "Set IMAGE_PATH to a valid image file, e.g.:"
  echo "  IMAGE_PATH=/path/to/image.jpg $0"
  exit 1
fi

echo "IMAGE_PATH=$IMAGE_PATH BASE_URL=$BASE_URL CONCURRENCY=$CONCURRENCY"
for i in $(seq 1 "$CONCURRENCY"); do
  curl -s -X POST "$BASE_URL/edit" \
    -F "image=@$IMAGE_PATH" \
    -F "prompt=Make the background blue" \
    -F "seed=$(( 1000 + i ))" \
    -F "steps=28" \
    --output "result_$i.png" &
done
wait
echo "Done. Check result_1.png ... result_$CONCURRENCY.png"
