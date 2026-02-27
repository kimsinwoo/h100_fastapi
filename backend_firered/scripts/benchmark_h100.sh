#!/usr/bin/env bash
# Benchmark POST /edit latency on H100. Measures time-to-first-byte and total duration.
# Usage: ./scripts/benchmark_h100.sh [path/to/image.png] [number of requests]
set -e
IMAGE_PATH="${1:-./test_image.png}"
N="${2:-5}"
BASE_URL="${BASE_URL:-http://localhost:8000}"
PROMPT="${PROMPT:-Make the background blue}"

if [ ! -f "$IMAGE_PATH" ]; then
  echo "Set IMAGE_PATH to an existing image file."
  exit 1
fi

echo "Benchmark: $N requests to $BASE_URL/edit (image=$IMAGE_PATH)"
echo "---"

TOTAL=0
for i in $(seq 1 "$N"); do
  START=$(date +%s.%N)
  CODE=$(curl -s -o /tmp/bench_result.png -w "%{http_code}" -X POST "$BASE_URL/edit" \
    -F "image=@$IMAGE_PATH" \
    -F "prompt=$PROMPT" \
    -F "seed=$i" \
    -F "steps=28")
  END=$(date +%s.%N)
  ELAPSED=$(echo "$END - $START" | bc)
  TOTAL=$(echo "$TOTAL + $ELAPSED" | bc)
  echo "Request $i: HTTP $CODE, ${ELAPSED}s"
  if [ "$CODE" != "200" ]; then
    echo "Failure body (first 300 chars):"
    head -c 300 /tmp/bench_result.png 2>/dev/null || true
    echo ""
  fi
done

AVG=$(echo "scale=2; $TOTAL / $N" | bc)
echo "---"
echo "Average latency: ${AVG}s over $N requests"
