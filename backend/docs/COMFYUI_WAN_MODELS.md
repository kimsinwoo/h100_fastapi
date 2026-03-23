# Wan 2.1 / VACE + Kijai ComfyUI 모델 배치 기준

이 문서는 **ComfyUI 쪽 디렉터리**에 Wan 관련 가중치를 두는 **표준 경로**를 정리합니다.

> **어디에 두나요?** `zimage_webapp` Git 저장소가 아니라, **실제로 ComfyUI 서버를 실행하는 설치 폴더** 의 `models/` 입니다.  
> **talktailForPet** 모노레포에서는 보통 레포 루트의 **`ComfyUI/models/`** 를 씁니다. 백엔드 연동·`.env` 예시는 레포 루트 **`COMFYUI.md`**, 스크립트 `bash scripts/print_comfyui_local_env.sh` 를 참고하세요.  
백엔드는 **내보낸 API 워크플로 JSON** 안의 파일명과 일치해야 하므로, 아래 이름으로 두거나 워크플로에서 선택한 이름을 이 구조에 맞추면 됩니다.

## Hugging Face (Kijai — LoRA 등)

- 컬렉션: **[Kijai/WanVideo_comfy](https://huggingface.co/Kijai/WanVideo_comfy)**  
- 예시 (14B CausVid LoRA):  
  [`Wan21_CausVid_14B_T2V_lora_rank32.safetensors`](https://huggingface.co/Kijai/WanVideo_comfy/blob/main/Wan21_CausVid_14B_T2V_lora_rank32.safetensors)

동일 리포지토리에서 1.3B용 LoRA 등도 받을 수 있습니다.

**Diffusion / VAE 단일 파일**은 용량이 커서 **[Comfy-Org/Wan_2.1_ComfyUI_repackaged](https://huggingface.co/Comfy-Org/Wan_2.1_ComfyUI_repackaged)** 의 `split_files/` 아래에 정리되어 있습니다. 워크플로가 기대하는 파일명(예: `wan2.1_vace_14B_fp16.safetensors`)과 ComfyUI `models/` 하위 폴더명이 **정확히 일치**해야 합니다.

### 「모델이 없습니다」가 뜰 때

ComfyUI가 목록에 없는 파일을 요구하면, 아래를 **실제 ComfyUI 루트** (예: `~/ComfyUI`) 기준으로 받습니다.

```bash
cd zimage_webapp/backend
pip install huggingface_hub

# LoRA + VAE (~242MB) + VACE 14B diffusion (~32GB) — 디스크·시간 여유 있을 때
python scripts/setup_wan_comfy_models.py ~/ComfyUI --vae --vace-14b -y
```

- **14B만** 필요하면: `--vace-14b -y` (LoRA는 이미 있다면 `--skip-loras --vace-14b -y`)
- **VAE만**: `--skip-loras --vae -y`
- **텍스트 인코더**(`umt5_xxl_*.safetensors`)도 없다면 같은 [Comfy-Org](https://huggingface.co/Comfy-Org/Wan_2.1_ComfyUI_repackaged/tree/main/split_files/text_encoders) 에서 `split_files/text_encoders/` 파일을 받아 `models/text_encoders/` 에 두면 됩니다.

스크립트는 HF 캐시에 받은 뒤 ComfyUI 쪽에는 가능하면 **하드링크**로 연결합니다(같은 디스크면 **추가 용량을 거의 쓰지 않음**). 복사(`copy`)는 다른 볼륨 등으로 링크가 안 될 때만 사용합니다.  
이전에 VACE 14B 복사 중 `No space left on device` 가 났다면 디스크를 비운 뒤 스크립트를 최신으로 하고 `--skip-loras --vace-14b -y` 만 다시 실행하면 됩니다(HF 캐시에 이미 받아 두었으면 재다운로드 없이 링크만 시도).

## ComfyUI 폴더 구조 (권장)

`ComfyUI` 루트를 `$COMFYUI` 로 둘 때:

```
ComfyUI/
├── models/
│   ├── diffusion_models/
│   │   ├── wan2.1_vace_14B_fp16.safetensors
│   │   └── wan2.1_vace_1.3B_fp16.safetensors
│   ├── text_encoders/
│   │   └── umt5_xxl_fp16.safetensors
│   │       또는 umt5_xxl_fp8_e4m3fn_scaled.safetensors  (둘 중 하나)
│   ├── loras/
│   │   ├── Wan21_CausVid_14B_T2V_lora_rank32.safetensors
│   │   └── Wan21_CausVid_bidirect2_T2V_1_3B_lora_rank32.safetensors
│   └── vae/
│       └── wan_2.1_vae.safetensors
├── input/
└── output/
```

### 역할 요약

| 구분 | 파일 예시 | 비고 |
|------|-----------|------|
| Diffusion | `wan2.1_vace_14B_fp16.safetensors` / `wan2.1_vace_1.3B_fp16.safetensors` | 14B: 480p·720p / 1.3B: 보통 480p만 |
| LoRA (가속·품질) | `Wan21_CausVid_14B_T2V_lora_rank32.safetensors` 등 | Kijai [WanVideo_comfy](https://huggingface.co/Kijai/WanVideo_comfy) |
| VAE | `wan_2.1_vae.safetensors` | Wan 2.1 워크플로와 짝 맞춤 |
| Text encoder | `umt5_xxl_*.safetensors` | fp16 또는 fp8 중 하나 (노드/래퍼 호환에 맞출 것) |

## 서버에서 적용 순서

1. **ComfyUI + Kijai Wan 커스텀 노드** 설치 (사용 중인 래퍼/README 기준).
2. 위 경로에 모델·LoRA·VAE·T5를 배치 (파일명은 워크플로 드롭다운과 동일하게).
3. ComfyUI에서 워크플로를 구성해 **한 번 끝까지 실행**해 확인.
4. **API 형식으로 저장**한 JSON을 `pipelines/` 등에 두고, 이 백엔드와 연동할 경우 **주입 규칙이 LTX용과 다를 수 있음** — Wan 전용 워크플로·설정이 필요하면 별도 이슈/브랜치에서 분기하는 것을 권장 (`docs/COMFYUI_FULL_SETUP.md` 참고).

## 자동 배치 스크립트

Kijai 리포에서 **아래 LoRA 두 개**를 `models/loras/` 로 내려받고, 나머지 표준 폴더를 만듭니다.

- `Wan21_CausVid_14B_T2V_lora_rank32.safetensors` ([HF](https://huggingface.co/Kijai/WanVideo_comfy/blob/main/Wan21_CausVid_14B_T2V_lora_rank32.safetensors))
- `Wan21_CausVid_bidirect2_T2V_1_3B_lora_rank32.safetensors`

```bash
cd zimage_webapp/backend
pip install huggingface_hub   # 최초 1회
# 아래 경로는 실제 ComfyUI 설치 루트로 바꿉니다 (문서용 예시 /path/to/ComfyUI 는 사용하지 마세요).
python scripts/setup_wan_comfy_models.py ~/ComfyUI
# diffusion + VAE 까지 한 번에 (용량 큼):
# python scripts/setup_wan_comfy_models.py ~/ComfyUI --vae --vace-14b -y
```

폴더만 만들고 HF 다운로드는 건너뛰려면:  
`python scripts/setup_wan_comfy_models.py ~/ComfyUI --skip-loras`

`text_encoders` 의 umt5 는 **수동**으로 Comfy-Org `split_files/text_encoders/` 에서 받거나, 스크립트로는 LoRA·VAE·VACE 14B 만 자동화되어 있습니다.

## 성능·타임아웃 참고

- 예: RTX 4090, 81프레임 720p 등 **수십 분** 걸릴 수 있음 → `.env` 의 `COMFYUI_TIMEOUT_SECONDS` / `COMFYUI_VIDEO_TIMEOUT_SECONDS` 를 충분히 크게 설정.
