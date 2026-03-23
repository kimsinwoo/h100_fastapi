# Dance pipelines (ComfyUI API format)

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

## Environment

- `DANCE_USE_SDXL_POSE_PIPELINE=true` — enable per-frame SDXL pose path (requires valid exported `dog_pose_generation.json` with ≥2 `LoadImage` nodes).
- `COMFYUI_BASE_URL`, `COMFYUI_ENABLED=true`
- `COMFYUI_REFERENCE_VIDEO_DIR` — when using LTX + reference video, align with ComfyUI input directory.
