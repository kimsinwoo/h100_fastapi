#!/usr/bin/env bash
# Example curl commands for SDXL Image Service (default: http://localhost:7000)
BASE="${BASE_URL:-http://localhost:7000}"

echo "GET $BASE/api/health"
curl -s "$BASE/api/health" | jq .

echo ""
echo "POST $BASE/api/generate (txt2img)"
curl -s -X POST "$BASE/api/generate" \
  -F "prompt=a photo of a cat on a sofa" \
  -F "style=realistic" \
  -F "steps=30" \
  -F "cfg=7.5" \
  -o /tmp/sdxl_out.png && echo "Saved /tmp/sdxl_out.png"

echo ""
echo "POST $BASE/api/generate (img2img, requires input image)"
# curl -s -X POST "$BASE/api/generate" \
#   -F "prompt=anime style" \
#   -F "style=anime" \
#   -F "image=@/path/to/input.png" \
#   -F "strength=0.75" \
#   -F "steps=30" \
#   -o /tmp/sdxl_img2img.png

echo ""
echo "POST $BASE/api/train-lora"
# curl -s -X POST "$BASE/api/train-lora" \
#   -H "Content-Type: application/json" \
#   -d '{"dataset_path": "./mydata", "output_path": "./lora_out", "rank": 4, "steps": 500}' | jq .
