# vLLM 서버 실행 방법

## 1. 한 줄 명령으로 바로 실행 (스크립트 없이, 로그 즉시 출력)

아래를 **그대로 복사**해 터미널에서 실행하면 vLLM이 바로 뜨고 로그가 즉시 찍힙니다.

```bash
cd ~/project/h100_fastapi/backend && source venv/bin/activate && PYTHONUNBUFFERED=1 vllm serve openai/gpt-oss-20b --port 7001 --host 0.0.0.0 --gpu-memory-utilization 0.85 --max-num-seqs 64 --max-model-len 32768 --dtype bfloat16 --quantization none --trust-remote-code --enforce-eager
```

- **같은 머신에서**: `cd` 경로만 실제 backend 경로로 바꾸면 됩니다.
- **기동만 빨리**: `--enforce-eager` 포함되어 있어 CUDA 그래프 캡처 없이 기동합니다.
- **양자화**: `--quantization none` (fp8 오류 나면 이렇게 실행).

## 2. 최소 스크립트로 실행

```bash
cd ~/project/h100_fastapi/backend
source venv/bin/activate
bash scripts/run_vllm_minimal.sh
```

`run_vllm_minimal.sh`는 위 한 줄과 동일한 옵션만 넣은 최소 스크립트라, 실행하면 곧바로 vLLM 로그가 나옵니다.  
(`chmod +x scripts/run_vllm_minimal.sh` 후 `./scripts/run_vllm_minimal.sh` 로 실행해도 됩니다.)

## 3. LLM 채팅 "All connection attempts failed" 해결

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
