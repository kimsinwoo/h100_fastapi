# Wan VACE 14B v2v 워크플로 (이미지 + 영상 → 영상)

백엔드에서 `WAN_VACE_V2V_ENABLED=true` 이고 레퍼런스 영상이 있으면 `run_image_to_video` 가 **LTX 대신** 이 워크플로를 사용합니다.

## 파일

- 기본: `video_wan_vace_14B_v2v.json` (API 포맷, ComfyUI에서 내보낸 그래프로 교체 가능)

## ComfyUI 요구사항

- ComfyUI-WanVideoWrapper (WanVaceToVideo, LoadVideo `file` 입력 등)
- 모델: `wan2.1_vace_14B_fp16`, `wan_2.1_vae`, `umt5_xxl_fp16`, LoRA CausVid 14B 등 — `docs/COMFYUI_WAN_MODELS.md` 참고
- `COMFYUI_REFERENCE_VIDEO_DIR` = ComfyUI `input` 과 동일 경로 권장 (레퍼런스 mp4 복사)

## 주입 규칙

- `LoadImage` → 업로드된 이미지 파일명  
- `LoadVideo` → `file` / `video` 등 키로 레퍼런스 mp4 파일명  
- `CLIPTextEncode` (id 순) → positive / negative  
- `KSampler` → `seed` (API에서 전달 시)  
- `WanVaceToVideo` → 선택적으로 `width` / `height` / `length` (num_frames)
