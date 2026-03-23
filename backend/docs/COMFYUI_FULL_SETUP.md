# ComfyUI 전체 세팅 (LTX i2v + 댄스 pose_sdxl)

이 문서는 **이 백엔드(`zimage_webapp/backend`)가 실제로 하는 주입(injection)** 과 일치하도록 작성했습니다.  
웹에서 흔히 보이는 예시 JSON의 `class_type` 이름(`LTXImageToVideo`, `ControlNetApply` 등)은 **설치한 커스텀 노드마다 다릅니다.** 반드시 **ComfyUI에서 워크플로를 만든 뒤「API 형식으로 저장」** 한 파일을 사용하세요.

## 아키텍처

```
Frontend (React) → FastAPI → ComfyUI (:8188) → mp4 / png
```

- **LTX (i2v)**: `LTX2_USE_COMFYUI=true` + `pipelines/<COMFYUI_LTX23_WORKFLOW>.json` (기본 `ltx23_i2v`)
- **댄스 pose_sdxl**: API `pipeline=pose_sdxl` + `pipelines/dance/dog_pose_generation.json` (운영 시 ComfyUI 내보내기로 교체)
- **Wan 2.1 / VACE (Kijai)**: 모델·LoRA 배치 경로는 **[`COMFYUI_WAN_MODELS.md`](COMFYUI_WAN_MODELS.md)** 참고 (LTX와 별도 워크플로).

---

## 1. ComfyUI 쪽 필수 구성

### 공통

| 항목 | 설명 |
|------|------|
| **ComfyUI** | 로컬 또는 Docker, 기본 포트 `8188` |
| **ffmpeg** | 영상 입출력·후처리에 필요 |
| **ComfyUI-Manager** | 커스텀 노드 설치·업데이트에 권장 |

### LTX (i2v) 파이프라인

| 항목 | 설명 |
|------|------|
| **Lightricks LTX Video** | 모델은 `ComfyUI/models/` 하위 적절한 폴더에 배치 (배포판 README 참고) |
| **ComfyUI-LTXVideo** (또는 Manager에서 제공하는 LTX 패키지) | 워크플로의 **노드 `class_type`이 설치본과 정확히 일치**해야 함 |
| **ComfyUI-VideoHelperSuite (VHS)** | 레퍼런스 영상 사용 시 `VHS_LoadVideo` 등 — 백엔드는 `video` / `file_path` / `path` / `filename` / `input` 키에 파일명 주입 시도 |

### pose_sdxl (댄스 프레임) 파이프라인

| 항목 | 설명 |
|------|------|
| **SDXL Base + VAE** | `checkpoints/`, `vae/` |
| **ControlNet** | OpenPose 또는 **DWpose** (권장) — `controlnet/` |
| **IPAdapter + CLIP Vision** | 정체성 유지 — `ipadapter/`, `clip_vision/` |
| **백엔드 MediaPipe** | 포즈 PNG는 FastAPI 쪽에서 생성 (이미 프로젝트에 연동) |

### 폴더 구조 (권장)

```
ComfyUI/
├── input/          ← 레퍼런스 mp4 복사 대상 (COMFYUI_REFERENCE_VIDEO_DIR 와 동일 경로 권장)
├── output/
├── models/
│   ├── checkpoints/
│   ├── controlnet/
│   ├── ipadapter/
│   ├── clip_vision/
│   └── vae/
└── custom_nodes/
    ├── ComfyUI-LTXVideo/          (또는 설치명에 맞는 이름)
    ├── ComfyUI-VideoHelperSuite/
    └── …
```

---

## 2. 백엔드 환경 변수 (핵심)

`.env` / `.env.example` 참고. 요약:

| 변수 | 역할 |
|------|------|
| `COMFYUI_ENABLED` | `true` 여야 ComfyUI 호출 경로 사용 |
| `COMFYUI_BASE_URL` | 예: `http://127.0.0.1:8188` |
| `COMFYUI_TIMEOUT_SECONDS` | LTX는 10~20분 걸릴 수 있음 → `900`~`1800` 권장 |
| `COMFYUI_VIDEO_TIMEOUT_SECONDS` | (선택) 영상 전용 상한 |
| `COMFYUI_REFERENCE_VIDEO_DIR` | **절대 경로**. ComfyUI `input` 과 맞추면 레퍼런스 mp4 복사 후 파일명만 워크플로에 주입 |
| `COMFYUI_OUTPUT_DIR` | (선택) ComfyUI 출력과 백엔드가 공유할 때 |
| `LTX2_USE_COMFYUI` | `true` 이면 LTX i2v 를 **diffusers 대신 ComfyUI** 로 |
| `COMFYUI_LTX23_WORKFLOW` | 확장자 없이 파일명 (기본 `ltx23_i2v` → `pipelines/ltx23_i2v.json`) |

레퍼런스 영상이 있는 요청은 `ltx23_i2v_ref.json` 이 있으면 우선 사용합니다. 자세한 내용은 `pipelines/README_LTX23.md` 를 보세요.

---

## 3. LTX 주입 규칙 (실제 코드 기준)

구현: `app/services/comfyui_service.py` 의 `_inject_ltx23_workflow_inputs`, `inject_reference_video`

1. **이미지**: `class_type`에 `loadimage` 가 포함된 노드의 `inputs.image` ← 업로드된 파일명  
2. **Positive / Negative**: `CLIP` + `Text` 가 들어간 노드들을 **노드 id 문자열 순으로 정렬** 후, 첫 번째=positive, 두 번째=negative  
3. **PrimitiveStringMultiline** 중 title 이 `Prompt` 인 것이 있으면 `value` 에 positive 프롬프트 주입 (CLIP 보조용)  
4. **레퍼런스 비디오**: `VHS_LoadVideo` 등 로더에 `video` / `file_path` / `path` / `filename` / `input` 중 존재하는 키로 파일명 주입  

⚠️ 워크플로에 **레퍼런스용 비디오 로더 노드가 없으면** 레퍼런스는 주입되지 않습니다. ComfyUI 그래프에 노드를 추가하고 API 포맷으로 다시 저장하세요.

---

## 4. pose_sdxl (댄스) 주입 규칙 (실제 코드 기준)

구현: `inject_dance_pose_workflow_inputs`

API에서는 **`pipeline=pose_sdxl`** 로 선택 (레거시 env `DANCE_USE_SDXL_POSE_PIPELINE` 은 더 이상 댄스 생성 경로에서 쓰이지 않음).

1. **`LoadImage`**: 노드 id 순 정렬 후  
   - **첫 번째** → 캐릭터(정체성 / IPAdapter 쪽)  
   - **두 번째** → 포즈 스켈레톤 PNG (ControlNet 쪽)  
2. **`CLIPTextEncode`**: 동일하게 id 순, 첫=positive, 둘째=negative  
3. **`KSampler` / `KSamplerAdvanced`**: `inputs` 에 `seed` 가 있으면 매 프레임 시드로 덮어씀 (`seed % 2**31`)  

⚠️ 최소 템플릿 `dog_pose_generation.json` 은 **연결성 검증용**입니다. 실사용은 **SDXL+ControlNet+IPAdapter 내보내기**로 교체하세요.

---

## 5. JSON 예시에 대한 주의

아래와 같은 **교육용 스케치**는 노드 타입이 실제 설치본과 다를 수 있습니다.

- `LTXImageToVideo` → 실제 패키지의 정확한 `class_type` 사용  
- `ControlNetApply` / `IPAdapterApply` → 보통 **전처리 + 적용** 노드가 여러 개로 쪼개져 있음  

**항상 ComfyUI「API 형식 저장」본**을 `pipelines/` 에 넣는 것이 안전합니다.

---

## 6. 검증 스크립트 (선택)

구조만 빠르게 점검할 때:

```bash
cd zimage_webapp/backend
python scripts/verify_comfyui_workflow.py pipelines/ltx23_i2v.json --mode ltx
python scripts/verify_comfyui_workflow.py pipelines/dance/dog_pose_generation.json --mode pose
# 내보낸 SDXL 그래프가 CLIP×2·KSampler(seed)까지 갖췄는지 엄격 검사:
python scripts/verify_comfyui_workflow.py pipelines/dance/my_export.json --mode pose --strict
```

통과해도 ComfyUI 서버·모델·노드 버전이 맞지 않으면 런타임에서 실패할 수 있습니다.

---

## 7. 권장 rollout 순서

1. **LTX i2v** 만 ComfyUI에서 끝까지 성공 (mp4 출력 확인)  
2. **pose_sdxl** 프레임 생성 (PNG)  
3. 필요 시 pose PNG 시퀀스 → **LTX** 또는 ffmpeg 로 영상화  

이 순서가 디버깅 비용 대비 효과가 가장 큽니다.
