# SDXL Image Service

Production-ready SDXL-based image generation: txt2img, img2img, LoRA training, dynamic LoRA loading. H100-optimized (bfloat16, TF32). Commercial-safe models: Stability AI SDXL Base 1.0, Animagine XL.

## Quick start

```bash
cd backend
python -m venv venv
source venv/bin/activate  # or venv\Scripts\activate on Windows
pip install -r requirements_sdxl.txt
cp .env.sdxl.example .env
uvicorn app.main_sdxl:app --host 0.0.0.0 --port 7000
```

## API

- `GET /api/health` — GPU and loaded models
- `POST /api/generate` — txt2img or img2img (multipart: prompt, style, optional image file)
- `POST /api/train-lora` — start LoRA training (JSON body)
- `GET /api/styles` — available style keys

## Example curl

**Health**
```bash
curl -s http://localhost:7000/api/health | jq
```

**Txt2img**
```bash
curl -X POST http://localhost:7000/api/generate \
  -F "prompt=a photo of a cat" \
  -F "style=realistic" \
  -F "steps=30" \
  -F "cfg=7.5" \
  --output out.png
```

**Img2img**
```bash
curl -X POST http://localhost:7000/api/generate \
  -F "prompt=anime style cat" \
  -F "style=anime" \
  -F "image=@input.png" \
  -F "strength=0.75" \
  -F "steps=30" \
  --output out.png
```

**Train LoRA** (dataset: `dataset/images/` + `dataset/captions.txt`, each line `filename\tcaption`)
```bash
curl -X POST http://localhost:7000/api/train-lora \
  -H "Content-Type: application/json" \
  -d '{"dataset_path": "./mydata", "output_path": "./lora_out", "rank": 4, "steps": 500}'
```

## Dataset format for LoRA

```
mydata/
  images/
    img1.png
    img2.png
  captions.txt
```

`captions.txt` — one line per image: `filename\tcaption` (tab-separated).

## Docker

```bash
docker build -f Dockerfile.sdxl -t sdxl-service .
docker run --gpus all -p 7000:7000 -v $(pwd)/data:/app/data sdxl-service
```

## License

Uses Stability AI SDXL 1.0 and Cagliostro Lab Animagine XL. Ensure your use complies with their licenses.
