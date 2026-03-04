# LTX-2 이미지→동영상 (Image-to-Video)

[Lightricks LTX-2](https://huggingface.co/Lightricks/LTX-2) / [Hugging Face Space (ltx-2-TURBO)](https://huggingface.co/spaces/alexnasa/ltx-2-TURBO) 기반으로, **사진 + 프롬프트**를 넣으면 동영상이 생성됩니다.

## 요구 사항

- Python 3.11+
- CUDA GPU (권장: 24GB+ VRAM, H100 등)
- `diffusers`에 LTX-2 파이프라인 포함 버전 (예: diffusers >= 0.37.0, 버전은 [diffusers 문서](https://huggingface.co/docs/diffusers/main/en/api/pipelines/ltx2) 참고)

## API

- `GET /api/video/presets` — 테스트용 프롬프트 프리셋 2종 반환
- `POST /api/video/generate` — multipart: `image`, `prompt` (필수), 선택: `preset`, `negative_prompt`, `num_frames`, `num_inference_steps`, `seed`

## 테스트 스타일 (프리셋)

1. **smile_turn**: 캐릭터가 웃으며 천천히 고개를 돌림
2. **wind_leaves**: 배경에서 나뭇잎이 바람에 흔들림

## 웹에서 사용

- 프론트엔드 `/video` 경로에서 이미지 업로드 → 프롬프트 입력 또는 위 2개 테스트 스타일 선택 → "동영상 생성" 클릭 후 재생/다운로드 가능.
