# Z-Image Backend

## 1) 메인 서버만 (포트 7000)

```bash
cd zimage_webapp/backend
python -m venv venv
source venv/bin/activate   # Windows: venv\Scripts\activate
pip install -r requirements.txt
uvicorn app.main:app --host 0.0.0.0 --port 7000
```

또는 `./run.sh` (의존성 설치 후 동일하게 7000번 실행)

- 접속: http://localhost:7000  
- API 문서: http://localhost:7000/docs  

---

## 2) LLM 서버(7001) + 메인(7000) — 다중 사용자

vLLM 공식 사용법과 동일하게 **pip 설치 → serve** 순서. 기본 포트는 8000이지만, 이 프로젝트는 메인 앱(7000)과 구분하기 위해 **7001**을 씁니다.

**터미널 1 — LLM (7001)**

```bash
cd zimage_webapp/backend
source venv/bin/activate
pip install vllm
export VLLM_PORT=7001
bash scripts/run_vllm_minimal.sh
```

- 기본 모델: `openai/gpt-oss-20b`  
- **Qwen3.5-35B-A3B** (공식: `vllm serve "Qwen/Qwen3.5-35B-A3B"`):
  ```bash
  pip install vllm   # 또는 최신 nightly: bash scripts/upgrade_vllm_for_qwen35.sh
  export VLLM_MODEL=Qwen/Qwen3.5-35B-A3B
  export VLLM_PORT=7001
  bash scripts/run_vllm_minimal.sh
  ```
  (스크립트가 `VLLM_USE_V1=0` 자동 적용 — RMSNormGated 오류 방지)

**OpenAI 호환 API 테스트 (curl, 포트 7001)**

```bash
curl -X POST "http://localhost:7001/v1/chat/completions" \
  -H "Content-Type: application/json" \
  --data '{
    "model": "Qwen/Qwen3.5-35B-A3B",
    "messages": [{"role": "user", "content": "Hello in one sentence."}]
  }'
```

이미지 입력(멀티모달) 예시:

```bash
curl -X POST "http://localhost:7001/v1/chat/completions" \
  -H "Content-Type: application/json" \
  --data '{
    "model": "Qwen/Qwen3.5-35B-A3B",
    "messages": [{
      "role": "user",
      "content": [
        {"type": "text", "text": "Describe this image in one sentence."},
        {"type": "image_url", "image_url": {"url": "https://cdn.britannica.com/61/93061-050-99147DCE/Statue-of-Liberty-Island-New-York-Bay.jpg"}}
      ]
    }]
  }'
```

**터미널 2 — 메인 (7000)**

```bash
cd zimage_webapp/backend
source venv/bin/activate
export LLM_USE_VLLM=true
uvicorn app.main:app --host 0.0.0.0 --port 7000
```

- 접속: http://localhost:7000 (LLM 요청은 7001로 전달)

---

## 환경변수 요약

| 변수 | 설명 |
|------|------|
| `VLLM_PORT` | LLM 서버 포트 (기본 7001) |
| `VLLM_MODEL` | vLLM 모델 (기본 gpt-oss-20b, Qwen3.5 시 위 참고) |
| `LLM_USE_VLLM` | true 시 메인 앱이 7001 vLLM 사용 |
| `VLLM_USE_V1` | Qwen3.5에서 오류 나면 0 (스크립트가 자동 설정) |

자세한 옵션: `docs/RUN_VLLM.md`, 프로젝트 루트 `RUN.md`
