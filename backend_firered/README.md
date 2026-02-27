# HunyuanImage-3.0-Instruct Backend

Production FastAPI backend for **tencent/HunyuanImage-3.0-Instruct**: single image generation/editing endpoint. H100-optimized with FlashInfer MoE.

## 1. Remove previous models

This codebase contains **only** HunyuanImage-3.0-Instruct. All prior editing/generation code (FireRed, SDXL, Flux, Qwen, etc.) and related config/schedulers have been removed.

## 2. Dependencies

- **requirements.txt**: torch 2.8, torchvision 0.23, torchaudio 2.8, transformers ≥4.35, flashinfer-python 0.5.0, FastAPI, uvicorn, pydantic v2, python-multipart, Pillow, safetensors.
- CUDA: use system CUDA 12.8 and install PyTorch from the cu128 index:
  ```bash
  pip install torch==2.8.0 torchvision==0.23.0 torchaudio==2.8.0 --index-url https://download.pytorch.org/whl/cu128
  pip install -r requirements.txt
  ```

## 3. Model integration (app/model.py)

- Load with `transformers.AutoModelForCausalLM.from_pretrained()`:
  - `attn_implementation="sdpa"`
  - `moe_impl="flashinfer"` for H100 MoE acceleration
  - `device_map="auto"`
  - `trust_remote_code=True`, `torch_dtype="auto"`, `moe_drop_tokens=True`
- Directory name must **not** contain dots (use e.g. `HunyuanImage-3-Instruct`).
- After loading: `model.load_tokenizer(model_id)`.

## 4. Image inference (app/model.py)

- `generate_image(prompt, image_paths, seed, image_size, use_system_prompt, bot_task, infer_align_image_size, diff_infer_steps, verbose, request_id)`:
  - **image_paths**: list of paths (max 3 for multi-image; /edit uses 1).
  - Defaults: steps=28, use_system_prompt="en_unified", infer_align_image_size=True, bot_task="think_recaption".
- Returns first generated PIL image; logging (inference time, GPU memory) done outside hot path.

## 5. FastAPI endpoint

- **POST /edit**
  - Body: `image` (file), `prompt` (string), optional `seed`, `steps`, `resolution`.
  - Validates upload, converts to temp file path, calls model inference, returns **PNG binary** via `StreamingResponse` (no base64).

## 6. Performance (H100)

- FlashInfer (`moe_impl="flashinfer"`) for MoE.
- No `.cpu()` or unnecessary transfers in hot path; batch size 1.
- First run may be slower due to FlashInfer kernel compilation.

## 7. Error handling

- JSON body: `{"error": "...", "message": "...", "request_id": "..."}`.
- 400: invalid image or prompt.
- 503: GPU OOM / inference failure.
- 504: timeout.

## 8. Logging

- Structured logs: prompt length, seed, num images, resolution, steps, inference duration, GPU memory before/after.

## 9. Dockerfile

- Base: CUDA 12.8.
- Installs dependencies, sets `MODEL_ID`, exposes 8000, runs uvicorn with uvloop.

## 10. Testing

**Curl example:**
```bash
curl -X POST -F "image=@img1.png" -F "prompt=Make the background blue" \
  -F "seed=42" -F "steps=28" \
  http://localhost:8000/edit --output result.png
```

**Scripts:**
- `scripts/curl_edit.sh` — single /edit call.
- `scripts/benchmark_h100.sh` — latency benchmark (multiple requests).

On failure, the response body is JSON (error/message/request_id).

## 11. Deployment

**Local (after downloading the model):**
```bash
# Download model (no dots in dir name)
huggingface-cli download tencent/HunyuanImage-3.0-Instruct --local-dir ./HunyuanImage-3-Instruct

export MODEL_ID=./HunyuanImage-3-Instruct
pip install -r requirements.txt
uvicorn app.main:app --host 0.0.0.0 --port 8000 --workers 1 --loop uvloop
```

**Docker:**
- Build image, mount or copy model into container at `/app/HunyuanImage-3-Instruct` (or set `MODEL_ID` accordingly).
- Run with GPU: `docker run --gpus all -p 8000:8000 ...`

**Distilled checkpoint (faster, 8 steps):**  
Use HunyuanImage-3.0-Instruct-Distil and set `MODEL_ID` to that path; reduce default steps (e.g. 8) in config if desired.
