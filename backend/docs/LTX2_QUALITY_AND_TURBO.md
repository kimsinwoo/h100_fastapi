# LTX-2 퀄리티 가이드 (ltx-2-TURBO 기준 정리)

이 문서는 **ltx-2-TURBO** (Lightricks 공식 Space/레포) 설정을 기준으로, 현재 FastAPI 백엔드의 LTX-2 Image-to-Video 품질을 맞추는 방법을 정리합니다.

---

## 1. 두 구현 방식 차이

| 항목 | ltx-2-TURBO (참조) | 현재 zimage_webapp (우리) |
|------|---------------------|----------------------------|
| **파이프라인** | Lightricks 네이티브 `DistilledPipeline` (ltx-pipelines, ltx-core) | HuggingFace **diffusers** `LTX2ImageToVideoPipeline` / `LTX2ConditionPipeline` |
| **모델 로딩** | safetensors 체크포인트 + Gemma 텍스트 인코더 + Spatial Upsampler + LoRA | `from_pretrained("Lightricks/LTX-2")` (diffusers 호환 체크포인트) |
| **스테이지** | **2단계**: Stage1 768×512 → Stage2 2× 업샘플 (1536×1024) | **1단계** (단일 해상도) |
| **스케줄** | Distilled sigma (고정 스텝 수, 8~9 스텝) | FlowMatchEulerDiscreteScheduler, num_inference_steps로 제어 |
| **해상도 기본** | 768×512 (1:1이면 512×512) | 640×384 (빠른 생성) / 품질 모드 시 768×512 |
| **프레임 수** | 121 (약 5초 @ 24fps) 또는 duration×24+1 | 33 (빠름) / 49, 81 (품질) |
| **스텝 수** | Stage1 distilled ~8스텝, Stage2 ~4스텝 | 10~40 (설정/API로 지정) |
| **Guidance scale** | 4.0 | 3.5 (빠름) / 4.0 (품질) |
| **Negative prompt** | 긴 시네마틱 품질용 문구 (아래 참고) | 짧은 문구 → **TURBO와 동일 문구로 통일 권장** |

---

## 2. ltx-2-TURBO에서 가져온 권장값 (constants.py 기준)

- **해상도 (품질)**  
  - Stage1: `height=512`, `width=768`  
  - 1:1이면 512×512
- **프레임**  
  - `DEFAULT_NUM_FRAMES = 121`, `DEFAULT_FRAME_RATE = 24.0`
- **스텝/가이던스**  
  - 비-distilled: `DEFAULT_NUM_INFERENCE_STEPS = 40`, `DEFAULT_CFG_GUIDANCE_SCALE = 4.0`
- **Negative prompt**  
  - TURBO `DEFAULT_NEGATIVE_PROMPT` (긴 문구) 사용 시 디테일·모션 품질에 유리합니다.  
  - 현재 `video_service`에는 **TURBO와 동일한 문구**를 `DEFAULT_NEGATIVE`(품질 모드)로 두었습니다.

---

## 3. 현재 백엔드에서 퀄리티 맞추기

### 3.1 설정(Config) / 환경변수

- **품질 우선**  
  - `ltx2_quality_mode=true` (또는 동일 역할 옵션)  
    → 해상도 768×512, guidance 4.0, TURBO negative prompt, steps 25~40, frames 49~81 권장.
- **속도 우선 (기본)**  
  - 해상도 640×384, steps 10, frames 33 유지.

### 3.2 API 호출 시

- `num_frames`: 49 또는 81 (8n+1)
- `num_inference_steps`: 25~40
- `negative_prompt`: 비워두면 서버 기본값(TURBO 스타일) 사용, 또는 클라이언트에서 TURBO 문구 전달.

### 3.3 Negative prompt (TURBO와 동일 문구)

`video_service`의 **품질용 기본 negative**는 ltx-2-TURBO `utils/constants.py`의 `DEFAULT_NEGATIVE_PROMPT`와 동일하게 맞춰 두었습니다.  
(blurry, out of focus, overexposed, … 부터 cinematic oversaturation, stylized filters, or AI artifacts. 까지)

---

## 4. ltx-2-TURBO 수준 품질을 완전히 쓰고 싶을 때

우리 백엔드는 **diffusers 1단계**만 사용하므로, TURBO의 **2단계(업샘플)** 및 **distilled sigma 스케줄**은 그대로 재현되지 않습니다.

- **옵션 A – 지금 스택으로 품질 극대화**  
  - 해상도 768×512, frames 81, steps 25~40, guidance 4.0, TURBO negative prompt 사용.  
  - 품질 모드 설정/API만 켜면 됨.

- **옵션 B – TURBO와 동일 파이프라인**  
  - `ltx-2-TURBO` 레포의 `DistilledPipeline` + ltx-pipelines/ltx-core를 그대로 사용하는 전용 서비스/스크립트를 두고,  
    FastAPI에서는 그 서비스를 호출하거나, TURBO 앱을 별도 실행해 두고 연동하는 방식.  
  - 이 경우 Gemma, 체크포인트, LoRA, spatial upsampler 등 TURBO와 동일한 의존성·경로 필요.

---

## 5. 파일별 정리

- **`app/services/video_service.py`**  
  - TURBO 기준 해상도/프레임/스텝/guidance/negative 상수 정리.  
  - 품질 모드 시 TURBO negative + 768×512, 49/81 frames, 25 steps, guidance 4.0 사용.
- **`app/core/config.py`**  
  - `ltx2_model_id`, `ltx2_use_full_cuda`, `ltx2_quality_mode`(또는 유사) 등으로 모드/품질 선택.
- **`app/api/routes.py`**  
  - `/api/video/generate`: `num_frames`, `num_inference_steps` 기본값을 품질 모드일 때 더 높게 적용 가능.
- **`README_LTX2.md`**  
  - 요구 사항, API, **진행 체크리스트** 링크.

---

## 6. 참고 경로 (ltx-2-TURBO)

- `ltx-2-TURBO/app.py` – Gradio UI, `generate_video()`, 파이프라인 호출
- `ltx-2-TURBO/ltx2_two_stage.py` – 2단계 CLI 예시
- `ltx-2-TURBO/packages/ltx-pipelines/src/ltx_pipelines/utils/constants.py` – 기본 해상도, 스텝, **negative prompt**
- `ltx-2-TURBO/packages/ltx-pipelines/src/ltx_pipelines/distilled.py` – `DistilledPipeline` Stage1/Stage2 로직
