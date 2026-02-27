# 실행 방법

- **메인 서버**: 포트 **7000** (이미지 생성, 채팅 API, 프론트 서빙)
- **LLM 서버**: 포트 **7001** (다중 사용자 채팅/프롬프트 추천 시 vLLM)

---

## 1) 메인 서버만 실행 (단일 사용자)

이미지 생성 + 채팅(로컬 LLM)을 **한 대에서** 쓸 때.

```bash
cd zimage_webapp/backend
pip install -r requirements.txt
source venv/bin/activate   # 가상환경 사용 시

# uvicorn으로 실행 (메인 서버 7000번)
uvicorn app.main:app --host 0.0.0.0 --port 7000
```

`run.sh` 를 쓰면 위와 동일하게 uvicorn을 실행합니다: `./run.sh`

- 접속: **http://localhost:7000**
- API 문서: http://localhost:7000/docs
- LLM은 **로컬 transformers**로 동작 (동시에 한 명만 권장)

---

## 2) 메인(7000) + LLM(7001) 실행 (다중 사용자)

여러 유저가 **동시에** 채팅을 쓰려면 **7001번에서 vLLM**을 띄웁니다.

**콘솔에 vLLM 로딩 나오고, 모델 불러온 뒤 채팅되는** 건 **7001에서 `vllm serve` 를 실행할 때**입니다.  
`uvicorn vllm_server.main:app --port 7001` 은 게이트웨이(프록시)만 띄우는 거라 **모델을 안 읽어서** 로딩 로그가 안 뜹니다.

**터미널 1: LLM 서버 (7001) — vllm serve (모델 로딩 콘솔에 뜸)**

```bash
cd zimage_webapp/backend
source venv/bin/activate
pip install vllm
export VLLM_PORT=7001
bash scripts/run_vllm_minimal.sh
```

- 기본 모델: **Qwen/Qwen3.5-35B-A3B** ([Hugging Face](https://huggingface.co/Qwen/Qwen3.5-35B-A3B)). 다른 모델은 `export VLLM_MODEL=openai/gpt-oss-20b` 등으로 지정.
- 콘솔에 vLLM 뜨고 모델 로딩 진행 → 완료되면 7001에서 채팅 가능.
- H100 등: `scripts/start_vllm_h100.sh` (자세한 옵션: `backend/docs/RUN_VLLM.md`)
- OOM 시: `VLLM_GPU_MEMORY_UTILIZATION=0.80` 또는 `VLLM_ENFORCE_EAGER=1`. Qwen3.5는 컨텍스트 262K 지원, OOM이면 `VLLM_MAX_MODEL_LEN=32768` 등으로 줄이기.

*`uvicorn vllm_server.main:app --port 7001` 은 게이트웨이만 띄우는 거라 **모델 로딩이 콘솔에 안 뜹니다.** 예전처럼 로딩 보이게 하려면 위처럼 **vllm serve** 를 7001에서 실행하세요.*

**터미널 2: 메인 서버 (7000)**

```bash
cd zimage_webapp/backend
source venv/bin/activate
pip install -r requirements.txt

# 메인 앱이 7001 LLM 서버를 쓰도록 설정 후 uvicorn 실행
export LLM_USE_VLLM=true
uvicorn app.main:app --host 0.0.0.0 --port 7000
```

- 접속: **http://localhost:7000**
- 메인은 7000, LLM 요청은 내부에서 **7001**로 전달됨

---

## 3) 프론트 개발 모드 (메인 7000 + Vite 3000)

백엔드만 7000에서 띄우고, 프론트는 Vite로 따로 띄울 때.

**터미널 1: 메인**

```bash
cd zimage_webapp/backend
source venv/bin/activate
uvicorn app.main:app --host 0.0.0.0 --port 7000
```

**터미널 2: 프론트**

```bash
cd zimage_webapp/frontend
npm install
npm run dev
```

- 프론트: http://localhost:3000 (Vite가 `/api`, `/static`, `/health` 를 7000으로 프록시)
- 백엔드: http://localhost:7000

---

## 4) 배포용 (프론트 빌드 후 메인이 서빙)

브라우저에서 **한 주소(7000)** 로만 접속하게 할 때.

```bash
cd zimage_webapp
./scripts/build_and_serve.sh   # frontend/dist → backend/static_frontend 복사
cd backend
source venv/bin/activate
uvicorn app.main:app --host 0.0.0.0 --port 7000
```

- 접속: **http://서버주소:7000**

다중 사용자 LLM이 필요하면 위 **2)** 처럼 7001에서 vLLM을 먼저 띄운 뒤, 메인 실행 시 `LLM_USE_VLLM=true` 로 기동하면 됩니다.
