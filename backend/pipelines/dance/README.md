# Dance pipelines (ComfyUI API format)

전체 설치·주입 규칙·환경 변수는 **[`docs/COMFYUI_FULL_SETUP.md`](../docs/COMFYUI_FULL_SETUP.md)** 를 참고하세요.

## `dog_pose_generation.json`

**Shipped file**: minimal **LoadImage → SaveImage** graph to verify ComfyUI connectivity (`/prompt` → `/history` → `/view`).

**Production**: Export your **SDXL + ControlNet (OpenPose/DWpose) + IPAdapter** graph from ComfyUI with **Save (API Format)** and replace this file. The backend injects:

- `LoadImage` #1 (sorted by node id): **character** reference (IPAdapter / identity)
- `LoadImage` #2: **pose skeleton** PNG (generated from MediaPipe keypoints)
- `CLIPTextEncode` (sorted): positive / negative
- `KSampler` / `KSamplerAdvanced`: `seed`

**IPAdapter vs LoRA (design)**:

- **IPAdapter**: image conditioning from a single reference photo — fast to set up, strong identity for short clips, no training.
- **LoRA**: subject-specific weights — best when you have trained a dog-identity LoRA; load it inside the same exported graph.

## `dog_video_generation.json`

Optional operator graph for **frame sequence → video** entirely inside ComfyUI (e.g. VHS + LTX).  
If omitted, the backend uses **ffmpeg** (`DanceGenerationService.generate_video`) or **LTX i2v** (`pipelines/comfyui_ltx23_workflow.json` / `ltx23_i2v.json`) via `run_image_to_video`.

## Environment / API

- **댄스 pose_sdxl 경로**: API `POST /api/dance/generate` (또는 custom)에서 **`pipeline=pose_sdxl`** 로 선택 (Form 필드).
- `COMFYUI_BASE_URL`, `COMFYUI_ENABLED=true`
- `COMFYUI_REFERENCE_VIDEO_DIR` — LTX + 레퍼런스 영상 시 ComfyUI `input` 과 동일 경로 권장.
- `DANCE_USE_SDXL_POSE_PIPELINE` — 레거시 플래그; **댄스 생성 경로에서는 사용하지 않음**. `pipeline=pose_sdxl` 사용 권장.
