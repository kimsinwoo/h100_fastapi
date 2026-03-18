# LTX-2.3-22b ComfyUI 워크플로

백엔드에서 `LTX2_USE_COMFYUI=true` 로 설정하면 이미지→비디오 생성이 ComfyUI(LTXVideo 노드)로 실행됩니다.

## 모델 위치

- LTX-2.3 체크포인트는 **ComfyUI/models** 에 두세요.
  - 예: `ComfyUI/models/diffusion_models/`, `ComfyUI/models/latent_upscale_models/` (upscaler 등)
- [Lightricks/LTX-2.3](https://huggingface.co/Lightricks/LTX-2.3) 에서 다운로드.

## 워크플로 파일

1. ComfyUI에서 **LTXVideo** 노드(ComfyUI Manager에서 설치)로 이미지→비디오 워크플로를 만듭니다.
2. **파일 → 저장** 으로 JSON으로 내보냅니다.
3. 이 디렉터리에 `ltx23_i2v.json` 이름으로 저장합니다.

설정에서 다른 이름을 쓰려면 `COMFYUI_LTX23_WORKFLOW=이름` (확장자 제외)으로 지정하세요.

## 댄스/레퍼런스 영상 반영 (선택)

커스텀 댄스 영상을 워크플로에 반영하려면:

1. ComfyUI에서 **Video Helper Suite** 설치 후 **VHS Load Video** 노드를 워크플로에 추가합니다.
2. 해당 노드의 **IMAGE** 출력을 포즈/ControlNet 등 레퍼런스를 받는 노드에 연결합니다.
3. **Save (API Format)** 으로 저장한 JSON을 **`ltx23_i2v_ref.json`** 이름으로 이 폴더에 둡니다.

레퍼런스 영상을 사용하는 요청이 들어오면 백엔드는 `ltx23_i2v_ref.json` 이 있으면 이를 사용하고, 없으면 기본 `ltx23_i2v.json` 만 사용합니다 (이 경우 레퍼런스 미반영 시 에러로 안내).

4. `.env` 에 **`COMFYUI_REFERENCE_VIDEO_DIR`** = ComfyUI의 input 폴더 절대 경로를 설정하면, 업로드/선택된 댄스 영상이 해당 폴더로 복사된 뒤 파일명이 워크플로에 주입됩니다.

## 성능·품질 (LTX-2.3 distilled)

- 해상도: 32 배수 (예: 768×512, 640×384).
- 프레임: 8n+1 (예: 33, 49, 121).
- distilled: 8 steps, CFG=1.0 권장.
- ComfyUI 출력 디렉터리를 백엔드와 공유하려면 `COMFYUI_OUTPUT_DIR` 에 절대 경로를 넣으세요.
