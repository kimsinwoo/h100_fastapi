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

vLLM 공식: `pip install vllm` 후 `vllm serve "Qwen/Qwen3.5-35B-A3B"` (기본 포트 8000). 여기서는 메인(7000)과 구분해 **7001** 사용.

```bash
cd zimage_webapp/backend
source venv/bin/activate
pip install vllm
export VLLM_PORT=7001
bash scripts/run_vllm_minimal.sh
```

- **기본 모델: openai/gpt-oss-20b** (vLLM 0.16에서 그대로 동작). 환경변수 없이 `bash scripts/run_vllm_minimal.sh` 만 실행하면 됩니다.
- **Qwen3.5-35B-A3B** 쓰려면 `qwen3_5_moe` 지원이 필요해, vLLM 0.16에서는 오류가 납니다. **한 번만** 업그레이드 스크립트 실행 후 모델 지정:
  ```bash
  cd zimage_webapp/backend && source venv/bin/activate
  bash scripts/upgrade_vllm_for_qwen35.sh
  export VLLM_MODEL=Qwen/Qwen3.5-35B-A3B
  export VLLM_PORT=7001
  bash scripts/run_vllm_minimal.sh
  ```
  - **"RMSNormGated has no attribute 'activation'"** 오류: vLLM 0.16.x nightly 버그. **한 번만** 패치 실행 후 재기동:
    ```bash
    cd zimage_webapp/backend && source venv/bin/activate
    python scripts/patch_vllm_rmsnorm_gated.py
    export VLLM_MODEL=Qwen/Qwen3.5-35B-A3B
    export VLLM_PORT=7001
    bash scripts/run_vllm_minimal.sh
    ```
    (`upgrade_vllm_for_qwen35.sh` 실행 시 위 패치를 자동 적용함)
- 콘솔에 vLLM 뜨고 모델 로딩 진행 → 완료되면 7001에서 채팅 가능.
- **"Port 7001 is already in use, trying port 7002"**: 같은 실행 안에서 API(7001)와 분산 통신이 같은 포트를 쓰면 발생함. 스크립트에서 분산 통신용으로 **API 포트+1**(7002)을 쓰도록 해 두었으므로, 재실행 시 API는 7001, 분산은 7002로 나뉘어 충돌하지 않음.
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

**7001 vLLM 쪽에 로그가 전혀 안 뜨는 경우**  
메인 앱과 vLLM이 **같은 호스트**에서 돌아야 기본 주소 `http://127.0.0.1:7001/v1` 로 연결됩니다. 메인과 vLLM을 **서로 다른 머신/팟**에서 띄우면, 메인은 자기 자신의 127.0.0.1:7001 로만 요청하므로 vLLM에는 요청이 가지 않습니다.  
→ **같은 노드에서 실행**하거나, 다른 호스트면 메인 서버에서 `export LLM_API_BASE=http://<vLLM이_도는_주소>:7001/v1` 로 vLLM이 실제로 떠 있는 주소를 지정하세요. 기동 시 로그에 `LLM (vLLM): 요청 주소=...` 가 찍히므로 그 주소가 vLLM에 도달 가능한지 확인하면 됩니다.

**vLLM 서버 직접 호출 (OpenAI 호환 API, 포트 7001)**

```bash
curl -X POST "http://localhost:7001/v1/chat/completions" \
  -H "Content-Type: application/json" \
  --data '{"model": "Qwen/Qwen3.5-35B-A3B", "messages": [{"role": "user", "content": "Hello."}]}'
```

이미지 입력 예시: [Hugging Face Qwen3.5-35B-A3B](https://huggingface.co/Qwen/Qwen3.5-35B-A3B) 의 vLLM 사용법 참고 (동일하게 `content`에 `image_url` 포함 가능).

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

---

## 5) 리버스 프록시 / Ingress 뒤에서 LLM 채팅 사용 시

채팅이 **스트리밍**으로 동작하며, 프록시가 응답을 버퍼링하면 브라우저에서 `ERR_INCOMPLETE_CHUNKED_ENCODING`(200 OK인데 본문이 끊김)이 날 수 있습니다. 이때 스트리밍 실패 후 대체로 비스트리밍 `/api/llm/chat` 를 호출하는데, 게이트웨이 타임아웃(504)이 나면 CORS 헤더가 없어서 "No 'Access-Control-Allow-Origin'" 로 보일 수 있습니다.

**필수 설정**

- **스트리밍 경로** (`/api/llm/chat/stream` 또는 `/{base_path}/api/llm/chat/stream`)  
  - **버퍼링 끄기**: nginx면 `proxy_buffering off;` (또는 백엔드에서 이미 `X-Accel-Buffering: no` 전송).  
  - **읽기 타임아웃**: LLM 응답이 길어질 수 있으므로 `proxy_read_timeout` 300초 정도 권장.
- **비스트리밍 LLM** (`/api/llm/chat`)  
  - 게이트웨이/Ingress의 **read timeout** 을 백엔드 LLM 타임아웃(기본 300초) 이상으로 설정.  
  - 504 등 **에러 응답에도 CORS 헤더**가 붙도록 Ingress/프록시 설정 (에러 시에도 `Access-Control-Allow-Origin` 등 전달).

**nginx 예시** (이 프로젝트 `frontend/nginx.conf` 에 스트림 전용 location 추가됨)

```nginx
location /api/llm/chat/stream {
    proxy_pass http://backend;
    proxy_http_version 1.1;
    proxy_set_header Host $host;
    proxy_buffering off;
    proxy_read_timeout 300s;
    # ... 나머지 proxy_set_header 등
}
```

Kubernetes Ingress 사용 시에는 해당 경로에 동일하게 버퍼링 비활성화·타임아웃 연장을 적용하고, 5xx 응답에도 CORS가 나가도록 설정하세요.
