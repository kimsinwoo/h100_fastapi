#!/usr/bin/env bash
# 설정 없이 바로 실행: 라이브러리 설치 후 서버 시작
# 사용법: cd backend && ./run.sh
set -e
cd "$(dirname "$0")"

if [ -d "venv" ]; then
  source venv/bin/activate
fi

# 첫 설치 시 torch/diffusers 등 대용량 패키지로 10~20분 걸릴 수 있음. 타임아웃·바이너리 우선으로 설치
PIP_TIMEOUT="${PIP_INSTALL_TIMEOUT:-900}"

# torch CUDA 버전 확인 후 CPU-only면 CUDA 버전으로 재설치 (H100 서버 전용)
TORCH_CUDA=$(python -c "import torch; print(torch.cuda.is_available())" 2>/dev/null || echo "False")
if [ "$TORCH_CUDA" = "False" ]; then
  echo "CUDA not detected in current torch. Installing CUDA-enabled torch (cu121)..."
  pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121 --timeout "$PIP_TIMEOUT" -q
fi

echo "Installing dependencies (timeout=${PIP_TIMEOUT}s, first run may take 10-20 min)..."
if [ -n "${PIP_VERBOSE:-}" ]; then
  pip install -r requirements.txt --timeout "$PIP_TIMEOUT" --prefer-binary
else
  pip install -r requirements.txt --timeout "$PIP_TIMEOUT" --prefer-binary -q
fi

PORT="${PORT:-7000}"
# 동영상 생성 등 장시간 요청: keep-alive 30분(1800초). 1분에 끊기면 uvicorn 앞단(리버스프록시/로드밸런서) read timeout을 600초 이상으로 올리세요.
# 7000 포트 사용 중이면: PORT=7001 ./run.sh 또는 .env에 PORT=7001
# 종료: Ctrl+C. 한 번에 안 멈추면 Ctrl+C 한 번 더 누르면 강제 종료.
echo "Starting Z-Image AI server on http://0.0.0.0:${PORT}"
exec uvicorn app.main:app --host 0.0.0.0 --port "$PORT" --timeout-keep-alive 1800
