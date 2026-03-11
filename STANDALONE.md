# 다른 서버에 이 레포만 clone 해서 사용하기

이 레포(**zimage_webapp**)는 상위 프로젝트(talktailForPet 등) 없이 **단독으로** 다른 서버에 배포해 사용할 수 있습니다.

## 1. Clone 및 실행

```bash
git clone <이 레포 URL> zimage_webapp
cd zimage_webapp
```

### 프론트 포함 한 번에 서빙 (권장)

```bash
./scripts/build_and_serve.sh   # 프론트 빌드 후 backend/static_frontend 로 복사
cd backend
cp .env.example .env            # 필요 시 .env 수정 (PORT, GPU 등)
./run.sh
```

브라우저: **http://서버주소:7000**

### API만 사용

```bash
cd backend
pip install -r requirements.txt
./run.sh
```

## 2. 의존성

- **이 레포만** 있으면 됩니다. 상위 폴더의 `shared`, `zit_lora_training` 등은 **필요 없습니다**.
- `shared/medical`(타입·테마): 이 레포 **안**에 `shared/medical` 이 포함되어 있어 프론트가 그대로 빌드됩니다.
- LoRA 학습: `backend/scripts/zit_lora/train_lora_zit.py` 가 이 레포에 포함되어 있어, 웹에서 학습 시작 시 별도 설치 없이 동작합니다.

## 3. 문서

- 실행 방법: [RUN.md](RUN.md)
- 배포·MIME 오류: [DEPLOY.md](DEPLOY.md)
- 백엔드 상세: [backend/README.md](backend/README.md)
