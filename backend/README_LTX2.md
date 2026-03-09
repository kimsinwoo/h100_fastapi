# LTX-2 이미지→동영상 (Image-to-Video)

[Lightricks LTX-2](https://huggingface.co/Lightricks/LTX-2) / [Hugging Face Space (ltx-2-TURBO)](https://huggingface.co/spaces/alexnasa/ltx-2-TURBO) 를 참고하여, **사진 + 프롬프트**로 동영상을 생성합니다.

- **구현**: HuggingFace **diffusers** (`LTX2ImageToVideoPipeline` / `LTX2ConditionPipeline`)
- **품질 기준**: 프로젝트 내 `ltx-2-TURBO` 레포의 상수·negative prompt를 맞춰 두었습니다. 자세한 비교와 옵션은 [docs/LTX2_QUALITY_AND_TURBO.md](docs/LTX2_QUALITY_AND_TURBO.md) 참고.

---

## 요구 사항

- Python 3.11+
- CUDA GPU (권장: 24GB+ VRAM, H100)
- **diffusers** (main 브랜치 권장):  
  `pip install git+https://github.com/huggingface/diffusers.git`
- **PyAV** (mp4 인코딩): `pip install av`
- (선택) **flash-attn**: `pip install flash-attn` → Flash Attention 2 사용 시 속도·품질 유리

---

## API

- `GET /api/video/presets` — 프롬프트 프리셋 2종
- `POST /api/video/generate` — multipart: `image`, `prompt` (필수), 선택: `preset`, `negative_prompt`, `num_frames`, `num_inference_steps`, `seed`

**품질 모드** (환경변수 `LTX2_QUALITY_MODE=true`):  
API에서 `num_frames`/`num_inference_steps`를 보내지 않으면 768×512, 49 frames, 25 steps, guidance 4.0 (ltx-2-TURBO 스타일) 적용.

---

## 테스트 프리셋

1. **smile_turn**: 캐릭터가 웃으며 천천히 고개를 돌림  
2. **wind_leaves**: 배경에서 나뭇잎이 바람에 흔들림  

---

## 진행해야 할 작업 (체크리스트)

아래는 LTX-2 영상 품질·안정화를 위해 **직접 진행하면 좋은 작업**입니다.

### 필수

- [ ] **diffusers 최신 main 설치**  
  `pip install -U "git+https://github.com/huggingface/diffusers.git"`
- [ ] **PyAV 설치**  
  `pip install av`
- [ ] **CUDA 환경 확인**  
  `python -c "import torch; print(torch.cuda.is_available())"`

### 품질·속도 (권장)

- [ ] **Flash Attention 2** (H100 등):  
  `pip install flash-attn`  
  → 설정 `enable_flash_attention_2=true` (기본값)로 로드 시 적용
- [ ] **품질 모드** 사용 시:  
  `.env`에 `LTX2_QUALITY_MODE=true`  
  → 768×512, 49 frames, 25 steps, TURBO 스타일 negative prompt 적용
- [ ] **VRAM 넉넉할 때**:  
  `LTX2_USE_FULL_CUDA=true` (기본) 유지 → CPU offload 없이 전체 GPU 사용

### 선택 (TURBO와 동일 스택 원할 때)

- [ ] **2단계·distilled 파이프라인**이 필요하면 `ltx-2-TURBO` 레포의 `DistilledPipeline` + ltx-pipelines/ltx-core를 별도 서비스로 띄우고, 필요 시 FastAPI에서 해당 서비스를 호출하도록 연동.  
  자세한 차이와 옵션: [docs/LTX2_QUALITY_AND_TURBO.md](docs/LTX2_QUALITY_AND_TURBO.md)

---

## 웹에서 사용

- 프론트엔드 `/video` 경로에서 이미지 업로드 → 프롬프트 또는 프리셋 선택 → "동영상 생성" 후 재생/다운로드.
