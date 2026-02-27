# FireRed-Image-Edit-1.0 Backend

Production-ready FastAPI backend for **FireRedTeam/FireRed-Image-Edit-1.0**. 기존과 동일하게 **uvicorn**으로 실행, 포트 **7000**. Single endpoint `POST /edit`: upload image + text prompt, get PNG stream.

## Project structure

```
backend_firered/
├── app/
│   ├── __init__.py
│   ├── main.py      # FastAPI app, POST /edit, lifespan, error handlers
│   ├── model.py     # Singleton pipeline load + run_edit (factory-friendly)
│   ├── schemas.py   # ErrorResponse, error_payload
│   ├── config.py    # Pydantic Settings (env)
│   └── utils.py     # load_image_rgb, request_id, gpu_memory_mb
├── requirements.txt
├── Dockerfile
├── README.md
├── run_local.sh
└── scripts/
    └── concurrency_test.sh
```

## Run locally

1. **Python 3.11**, CUDA, and pip:

   ```bash
   cd zimage_webapp/backend_firered
   python3.11 -m venv venv
   source venv/bin/activate   # Windows: venv\Scripts\activate
   pip install -r requirements.txt
   ```

2. **Optional env** (create `.env` or export):

   - `MODEL_ID` — default `FireRedTeam/FireRed-Image-Edit-1.0`
   - `MAX_CONCURRENT_JOBS` — default `2`
   - `TIMEOUT_SECONDS` — default `120`
   - `DEFAULT_STEPS` — default `28`
   - `DEFAULT_GUIDANCE` — default `7.0`

3. **Start server** (기존처럼 uvicorn, 포트 7000):

   ```bash
   uvicorn app.main:app --host 0.0.0.0 --port 7000 --workers 1 --loop uvloop
   ```

   Or use the helper script:

   ```bash
   chmod +x run_local.sh
   ./run_local.sh
   ```

## Run with Docker

1. **Build** (from repo root or from `backend_firered`):

   ```bash
   cd zimage_webapp/backend_firered
   docker build -t firered-backend .
   ```

2. **Run** (기존처럼 포트 7000, uvicorn, GPU):

   ```bash
   docker run --gpus all -p 7000:7000 firered-backend
   ```

   With env:

   ```bash
   docker run --gpus all -p 7000:7000 \
     -e MAX_CONCURRENT_JOBS=2 \
     -e TIMEOUT_SECONDS=120 \
     firered-backend
   ```

## Example curl

**POST /edit** — image file + form fields; response is raw PNG.

   ```bash
   curl -X POST "http://localhost:7000/edit" \
  -F "image=@/path/to/your/image.jpg" \
  -F "prompt=Add a red hat on the dog" \
  -F "seed=42" \
  -F "guidance_scale=7.0" \
  -F "steps=28" \
  --output result.png
```

- `image`: required (file)
- `prompt`: required (string)
- `seed`: optional (int)
- `guidance_scale`: optional (float, default from config)
- `steps`: optional (int, default from config)

Response: `image/png` binary. On error: JSON `{"error":"...","detail":"...","request_id":"..."}` with status 400 / 503 / 504.

## Concurrency test script

From project root:

```bash
chmod +x zimage_webapp/backend_firered/scripts/concurrency_test.sh
zimage_webapp/backend_firered/scripts/concurrency_test.sh
```

Uses a sample image path and sends multiple concurrent requests; adjust `IMAGE_PATH` and `BASE_URL` inside the script if needed.

## Errors (JSON)

| Status | error (example) | When |
|--------|------------------|------|
| 400 | invalid_file / invalid_image | Wrong file type or unreadable image |
| 503 | gpu_oom / inference_error | GPU OOM or runtime error |
| 504 | timeout | Queue or inference timeout (e.g. 120s) |

All responses include `request_id` for logging.

## Extensibility

- **model.py**: `load_pipeline(model_id=...)` supports passing a different HuggingFace ID (e.g. future FireRed-Image-Edit-1.0-Distilled). LoRA loading can be added in the same module without changing **main.py** or the `/edit` contract.
