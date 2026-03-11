# Z-Image Backend

FastAPI 서버 (포트 7000). 이미지 생성·비디오·ComfyUI 연동·LLM 채팅 등.

## 실행

(레포 루트에서 `backend` 로 이동 후)

```bash
cd backend
./run.sh
```

또는:

```bash
python -m venv venv
source venv/bin/activate   # Windows: venv\Scripts\activate
pip install -r requirements.txt
uvicorn app.main:app --host 0.0.0.0 --port 7000
```

- 접속: http://localhost:7000
- API 문서: http://localhost:7000/docs

포트 변경: `PORT=8000 ./run.sh`

## LLM (vLLM, 7001)

다중 사용자용 LLM은 별도 터미널에서:

```bash
source venv/bin/activate
export VLLM_PORT=7001
bash scripts/run_vllm_minimal.sh
```

메인 앱에서 vLLM 사용: `LLM_USE_VLLM=true` 로 실행.

## ComfyUI

로컬 ComfyUI 서버(기본 8188) 연동: `GET /api/comfyui/health`, `POST /api/comfyui/run`. 워크플로우 JSON은 `pipelines/` 에 두고 `pipeline_name` 으로 실행 가능.

## LTX-2.3-22b 이미지→비디오

- 기본 모델 ID: `Lightricks/LTX-2.3` (distilled: 8 steps, CFG=1.0).
- diffusers에 2.3 미지원 시: `LTX2_USE_COMFYUI=true` 로 두고 ComfyUI(LTXVideo 노드) 사용. 모델은 ComfyUI/models, 워크플로는 `pipelines/ltx23_i2v.json` (ComfyUI에서 내보낸 JSON). 자세한 내용은 `pipelines/README_LTX23.md` 참고.

## 환경변수

`.env.example` 참고. 주요 항목: `PORT`, `COMFYUI_BASE_URL`, `LLM_USE_VLLM`, `LLM_API_BASE`.
