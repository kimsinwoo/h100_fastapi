# Z-Image Backend

FastAPI 서버 (포트 7000). 이미지 생성·비디오·ComfyUI 연동·LLM 채팅 등.

## 실행

(레포 루트에서 `backend` 로 이동 후)

```bash
cd backend
./run.sh
```

**pip 설치가 오래 걸리거나 멈춘 것처럼 보일 때**  
첫 설치 시 torch/diffusers 등으로 10~20분 걸릴 수 있습니다. 진행 로그를 보려면:

```bash
cd backend
source venv/bin/activate   # 또는 가상환경 활성화
PIP_VERBOSE=1 ./run.sh
```

또는 설치만 먼저 (타임아웃 15분):

```bash
pip install -r requirements.txt --timeout 900 --prefer-binary -v
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
- **LTX-2.3은 diffusers에서 미지원**이므로 기본값으로 ComfyUI 경로 사용(`LTX2_USE_COMFYUI=true` 기본). 다음이 필요합니다:
  1. ComfyUI 서버 실행 (기본 8188), LTXVideo 노드 설치
  2. ComfyUI에서 이미지→비디오 워크플로를 JSON으로 내보낸 뒤 `pipelines/ltx23_i2v.json` 으로 저장
  3. LTX-2.3 모델은 ComfyUI의 `models/` 에 배치 (예: diffusion_models)
- 자세한 내용은 `pipelines/README_LTX23.md` 참고.

## 환경변수

`.env.example` 참고. 주요 항목: `PORT`, `COMFYUI_BASE_URL`, `LLM_USE_VLLM`, `LLM_API_BASE`.
