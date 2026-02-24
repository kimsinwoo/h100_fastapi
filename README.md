# Z-Image AI Web Service

Production-ready Image-to-Image AI web app: **FastAPI** backend with **Z-Image-Turbo**, **React + Vite + TypeScript** frontend.

## Features

- Upload image, choose from 10 style presets or add a custom prompt
- Z-Image-Turbo model loaded once at startup (singleton), GPU/CPU with autocast
- Async API, rate limiting, logging middleware, CORS
- Strict TypeScript frontend: drag & drop upload, style grid, strength slider, download result

## Folder structure

```
zimage_webapp/
├── backend/
│   ├── app/
│   │   ├── main.py           # FastAPI app, lifespan, middleware
│   │   ├── api/
│   │   │   └── routes.py     # POST /api/generate, GET /api/styles
│   │   ├── core/
│   │   │   └── config.py    # Settings from env
│   │   ├── services/
│   │   │   └── image_service.py  # Z-Image-Turbo singleton, inference
│   │   ├── schemas/
│   │   │   └── image_schema.py
│   │   ├── utils/
│   │   │   └── file_handler.py
│   │   └── models/
│   │       └── style_presets.py   # 10 style prompt templates
│   ├── static/
│   │   └── generated/       # Saved originals + generated images
│   ├── requirements.txt
│   ├── Dockerfile
│   └── .env.example
├── frontend/
│   ├── src/
│   │   ├── components/      # ImageUploader, StyleSelector, ResultViewer, LoadingSpinner
│   │   ├── services/api.ts
│   │   ├── types/api.ts
│   │   ├── App.tsx
│   │   └── main.tsx
│   ├── package.json
│   ├── Dockerfile
│   └── nginx.conf
├── docker-compose.yml
└── README.md
```

## LoRA 학습 (Z-Image-Turbo)

웹에서 **학습 데이터**(이미지 + 캡션)를 올린 뒤 **POST /api/training/start** 를 호출하면, 백엔드가 백그라운드에서 **Z-Image-Turbo LoRA** 학습을 돌립니다.

- **데이터**: `/api/training/items` 로 이미지·캡션·카테고리 추가. `POST /api/training/start` 시 `data/training/` 의 데이터를 `prepared_for_training/` (png+txt) 로 준비한 뒤, **talktailForPet/zit_lora_training** 의 `train_lora_zit.py` 를 `--dataset_dir` / `--output_dir` 로 실행합니다.
- **저장 위치**: 학습이 끝나면 LoRA 가 **backend/data/lora** (또는 설정의 `lora_adapters_dir`) 에 safetensors 로 저장됩니다.
- **필요 조건**: 프로젝트 루트에 **zit_lora_training** 폴더가 있어야 합니다 (같은 저장소의 `zit_lora_training`). 없으면 `LORA_TRAIN_CMD` 환경변수로 다른 학습 스크립트를 지정할 수 있습니다.

## Run instructions

### AI 서버 하나로 실행 (프론트 포함, MIME 오류 방지)

브라우저에서 **AI 서버 주소 하나**로 쓰려면, **빌드된 프론트**를 백엔드가 서빙해야 합니다. 그래야 JS 요청에 HTML이 아닌 실제 `.js`가 가서 "Expected a JavaScript module but got text/html" 오류가 나지 않습니다.

**1) 프론트 빌드 후 백엔드로 복사 (한 번만 실행)**

```bash
cd zimage_webapp
chmod +x scripts/build_and_serve.sh
./scripts/build_and_serve.sh
```

**2) 백엔드 실행**

```bash
cd backend
./run.sh
```

이후 브라우저에서 **http://서버:8000** 으로 접속하면 됩니다. (프론트는 Vite로 따로 띄우지 않아도 됩니다.)

**배포 후에도 "Expected JavaScript but got text/html" 오류가 나면:** 서버에 **빌드 결과물**(`frontend/dist` 안의 내용)만 올렸는지 확인하세요. 자세한 내용은 [DEPLOY.md](DEPLOY.md) 참고.

### AI 서버만 실행 (API만 쓸 때)

**.env 없이** 라이브러리만 설치하고 실행하면 됩니다.

```bash
cd backend
./run.sh
```

Windows:

```bash
cd backend
run.bat
```

또는 한 줄로:

```bash
cd backend && pip install -r requirements.txt && uvicorn app.main:app --host 0.0.0.0 --port 8000
```

- 서버: http://0.0.0.0:8000 (API 문서: http://localhost:8000/docs)
- 첫 요청 시 Hugging Face에서 모델 자동 다운로드 (수 GB). GPU 있으면 자동 사용, 없으면 CPU로 동작

### Local (venv 사용 시)

**Backend (Python 3.11+)**

```bash
cd backend
python -m venv venv
source venv/bin/activate   # Windows: venv\Scripts\activate
pip install -r requirements.txt
uvicorn app.main:app --host 0.0.0.0 --port 8000
```

- GPU 사용: `pip install torch --index-url https://download.pytorch.org/whl/cu121` 후 위 명령 재실행

**Frontend**

```bash
cd frontend
npm install
npm run dev
```

- Opens at http://localhost:3000; Vite proxies `/api` and `/static` to the backend (port 8000).

### Docker

**Backend only**

```bash
cd backend
docker build -t zimage-backend .
docker run -p 8000:8000 --gpus all zimage-backend
```

**Full stack (backend + frontend)**

```bash
docker compose up --build
```

- Frontend: http://localhost:80  
- Backend API: http://localhost:8000 (used by nginx proxy from frontend)

## API

- **POST /api/generate** — multipart: `file`, `style`, `custom_prompt` (optional), `strength` (optional), `seed` (optional). Returns `{ original_url, generated_url, processing_time }`.
- **GET /api/styles** — list of style keys and prompt descriptions.
- **GET /health** — `{ status, gpu_available }`.

## Production deployment notes

1. **Secrets**: Do not commit `.env`. Set `ENVIRONMENT=production`, strong CORS origins.
2. **Rate limiting**: Backend uses in-memory rate limit. For multi-worker, use Redis and share state.
3. **Static files**: Generated images live under `static/generated`. Ensure the volume is persistent and backups if needed.
4. **GPU**: Use `TORCH_INDEX=https://download.pytorch.org/whl/cu121` (or cu124) in Docker build for CUDA. Runtime: `--gpus all` or deploy on a GPU node.
5. **Model**: First run downloads `Tongyi-MAI/Z-Image-Turbo` from Hugging Face. Ensure network and disk space (~10GB+).
