# vLLM 서버 실행 방법

**실행 경로**: 프로젝트의 **backend** 디렉터리에서 실행합니다.  
- 이 저장소 기준: `zimage_webapp/backend`  
- 서버 예: `~/project/h100_fastapi/backend` 등 실제 backend 경로

**Linux + H100 NVL (1.19.210 등)**  
- 스크립트는 Linux·H100 NVL 환경에 맞춰 기본값을 사용합니다 (`--host 0.0.0.0`, `bfloat16`, `gpu-memory-utilization 0.88`, `max-num-seqs 96`).
- 실행 전 **nvidia-smi**로 GPU 사용 중인 프로세스가 없어야 합니다. OOM이 나면 `VLLM_ENFORCE_EAGER=1` 또는 `VLLM_GPU_MEMORY_UTILIZATION=0.80`, `VLLM_MAX_NUM_SEQS=32`로 조정하세요.
- H100 NVL 2-GPU 구성이면 `VLLM_TENSOR_PARALLEL_SIZE=2`로 두고 `scripts/start_vllm_h100.sh`를 사용하세요.

| 환경변수 | 기본값 | 설명 |
|----------|--------|------|
| `VLLM_GPU_MEMORY_UTILIZATION` | 0.88 | GPU 메모리 사용 비율. OOM 시 0.80 등으로 낮추기 |
| `VLLM_MAX_NUM_SEQS` | 96 | 동시 처리 시퀀스 수(연속 배치). OOM 시 32 등 |
| `VLLM_MAX_MODEL_LEN` | 32768 | 최대 컨텍스트 길이 |
| `VLLM_ENFORCE_EAGER` | 0 | 1이면 CUDA 그래프 비활성화(메모리 절약, 기동 빠름) |
| `VLLM_TENSOR_PARALLEL_SIZE` | 1 | 2-GPU NVL이면 2 |
| `VLLM_QUANTIZATION` | (비움) | gpt-oss-20b는 모델이 mxfp4 → 비우면 모델 기본값 사용. vLLM 0.15+ 필요 |
| `VLLM_PORT` | 7001 | 서버 포트 |

---

## 0. vLLM 엔진 없이 테스트 (Mock 모드)

**GPU·vLLM 설치 없이** `vllm_server` 게이트웨이만 띄워서 API 흐름을 검증할 수 있습니다.  
채팅 요청 시 실제 추론 대신 Mock 응답이 반환됩니다.

```bash
cd zimage_webapp/backend
source venv/bin/activate
export VLLM_USE_MOCK=1
python run_vllm_gateway.py
```

또는 스크립트로 (기본 포트 7001):

```bash
cd zimage_webapp/backend
source venv/bin/activate
bash scripts/run_vllm_gateway_mock.sh
```

**동작 검증** (다른 터미널에서):

```bash
cd zimage_webapp/backend
source venv/bin/activate
python scripts/test_vllm_gateway_mock.py
```

성공 시 `All checks passed.` 가 출력됩니다. 게이트웨이 기본 포트는 **7001**이라 메인 앱 `LLM_API_BASE=http://127.0.0.1:7001/v1` 그대로 쓰면 됩니다.

---

## 1. 한 줄 명령으로 바로 실행 (Linux, H100 NVL)

backend 디렉터리로 이동한 뒤, 아래 한 줄을 복사해 실행합니다.

```bash
cd zimage_webapp/backend && source venv/bin/activate && PYTHONUNBUFFERED=1 vllm serve openai/gpt-oss-20b --port 7001 --host 0.0.0.0 --gpu-memory-utilization 0.88 --max-num-seqs 96 --max-model-len 32768 --dtype bfloat16 --trust-remote-code --enforce-eager
```

서버 경로가 다르면 `cd`만 수정. 예: `cd ~/project/h100_fastapi/backend && ...`

- **H100 NVL**: `0.88` / `96` 은 80GB 기준. OOM 시 `--gpu-memory-utilization 0.80 --max-num-seqs 32` 로 낮추세요.
- **기동만 빨리**: `--enforce-eager` 포함.
- **mxfp4 오류**: `pip install -U "vllm>=0.15"` 후 재실행.

## 2. 최소 스크립트로 실행

```bash
cd zimage_webapp/backend
source venv/bin/activate
bash scripts/run_vllm_minimal.sh
```

`run_vllm_minimal.sh`는 위 한 줄과 동일한 옵션만 넣은 최소 스크립트라, 실행하면 곧바로 vLLM 로그가 나옵니다.  
(`chmod +x scripts/run_vllm_minimal.sh` 후 `./scripts/run_vllm_minimal.sh` 로 실행해도 됩니다.)

## 3. 오류 해결: `cannot import name 'default_cache_dir' from 'triton.runtime.cache'`

vLLM 0.7.x는 Triton 3.x와 API 호환이 되지 않아 이 오류가 날 수 있습니다. **아래 중 하나**를 적용한 뒤 다시 실행하세요.

**방법 A – Triton 2.x 사용 (vLLM 0.7 유지)**  
```bash
source venv/bin/activate
pip install 'triton>=2.0,<3.0'
# 그다음 1번 또는 2번으로 vLLM 서버 다시 실행
```

**방법 B – vLLM 업그레이드 (Triton 최신과 호환)**  
```bash
source venv/bin/activate
pip install -U "vllm>=0.15"
# 그다음 1번 또는 2번으로 vLLM 서버 다시 실행
```

`requirements-vllm.txt`에는 Triton 2.x 고정이 들어 있어, `pip install -r requirements-vllm.txt`로 재설치해도 동일하게 맞출 수 있습니다.

## 4. 오류 해결: `Quantization method specified in the model config (mxfp4) does not match ... (fp8)`

**openai/gpt-oss-20b**는 모델 설정이 **mxfp4**라서, `--quantization fp8`을 넘기면 위 오류가 납니다.

**해결:** vLLM 0.15 이상을 쓰고, **양자화 인자를 넘기지 않기** (모델 기본값 mxfp4 사용).

```bash
source venv/bin/activate
pip install -U "vllm>=0.15"
# VLLM_QUANTIZATION 은 설정하지 않음(비움). 그다음 스크립트 실행
./scripts/start_vllm_h100.sh
```

스크립트 기본값은 이제 `VLLM_QUANTIZATION`을 비워 두어 `--quantization`을 전달하지 않습니다. vLLM 0.15+에서만 mxfp4가 지원되므로, 0.7.x 환경이라면 반드시 위처럼 업그레이드한 뒤 실행하세요.

## 5. 오류 해결: `World size (2) is larger than the number of available GPUs (1)`

**GPU 1대**인데 `VLLM_TENSOR_PARALLEL_SIZE=2`로 두면 발생합니다. `start_vllm_h100.sh`는 이제 **GPU 개수를 자동으로 확인**해 `tensor_parallel_size`를 GPU 수 이하로 맞춥니다. 별도 설정 없이 다시 실행하면 됩니다.

수동으로 1대로 고정하려면:
```bash
export VLLM_TENSOR_PARALLEL_SIZE=1
./scripts/start_vllm_h100.sh
```

## 6. LLM 채팅 "All connection attempts failed" 해결

이 오류는 **메인 앱이 vLLM 서버에 연결하지 못할 때** 납니다.

- **vLLM이 같은 호스트에서 7001로 떠 있는 경우**  
  메인 앱 `.env` 또는 환경 변수:
  ```bash
  LLM_USE_LOCAL=false
  LLM_API_BASE=http://127.0.0.1:7001/v1
  ```
- **vLLM이 다른 Pod/호스트에 있는 경우**  
  메인 앱이 접속할 수 있는 주소로 설정:
  ```bash
  LLM_API_BASE=http://vLLM서버IP또는서비스명:7001/v1
  ```
- **vLLM이 아직 안 떠 있으면**  
  먼저 위 1번 또는 2번으로 vLLM을 띄운 뒤, 메인 앱을 재기동하세요.

요청 URL은 실패 시 앱 로그에 `URL=...` 로 남습니다. 그 URL이 실제 vLLM 주소와 같은지 확인하면 됩니다.
