# 목표(강아지 인식·댄스) vs 현재 구현 — 정리

## 1. ComfyUI 단독 실행 로그 (Wan 등) 해석

서버에서 `Prompt executed in 452s`, 진행률 100%면 **해당 워크플로는 정상 완료**입니다.

| 로그 | 의미 |
|------|------|
| `cu130 or higher` | 일부 CUDA 최적화 권장. **필수 아님.** |
| `sageattention` 없음 | 선택 패키지. 없어도 동작. |
| `deprecated legacy API` | 커스텀 노드 구버전 힌트. **실패 원인 아님.** |

**주의:** ComfyUI에서 Wan으로 돌리는 것과, **이 레포의 웹앱(zimage) 댄스 API**는 **별도 경로**입니다. 앱은 기본적으로 **LTX / pose_sdxl + 백엔드 파이프라인**을 씁니다.

---

## 2. 원하시는 동작에 맞게 쓰려면

| 목표 | 현재 zimage 백엔드 | 보강 방향 |
|------|-------------------|-----------|
| **강아지 “인식”** (이미지에서 자동 검출) | 없음. 사용자가 **강아지 사진을 업로드**하고 UI에서 dog/cat만 선택 | 선택: YOLO-pet / SAM 세그멘테이션 API 추가, 크롭 후 생성 |
| **사람 ↔ 강아지 “대치”** (영상 속 사람을 강아지로 바꿈) | **직접적인 인물 스왑/인페인팅 없음** | 레퍼런스는 **모션 참고**용. 완전한 “사람 제거 후 강아지 합성”은 별도 파이프라인(세그멘테이션+합성) 필요 |
| **댄스 모션** | **① LTX + 레퍼런스 영상** (영상 파일 주입), **② pose_sdxl** (영상→포즈→프레임 생성) | 레퍼런스가 **사람 댄스**면, 포즈는 **MediaPipe 인체** 기준 → **네 발 보행 강아지**와 어긋날 수 있음. 동물/사람 포즈 모델·수동 키프레임 검토 |

---

## 3. 코드 상 실제 경로 (요약)

- `DanceGenerationService.generate_dance_video` — `pipeline=ltx` | `pose_sdxl`
- **ltx**: `run_image_to_video` + (가능하면) **레퍼런스 mp4** → LTX/ComfyUI `ltx23_i2v` 계열
- **pose_sdxl**: `extract_poses_from_video` (**MediaPipe 인체 포즈**) → 스켈레톤 PNG → ComfyUI `dog_pose_generation.json` (SDXL+ControlNet+IPAdapter 내보내기 권장)

프론트: `VideoPage.tsx` — `dancePipeline` / `customDancePipeline`으로 API에 전달.

---

## 4. Wan VACE v2v (이미지 + 레퍼런스 영상 → 영상)

- 백엔드: `WAN_VACE_V2V_ENABLED=true` + `pipelines/video_wan_vace_14B_v2v.json` + `COMFYUI_ENABLED=true` 이면, **레퍼런스 영상이 있는** `run_image_to_video` 요청이 **LTX 대신** Wan VACE 워크플로를 탑니다 (`pipelines/README_WAN_VACE_V2V.md`).
- ComfyUI에 WanVideoWrapper·모델 파일이 있어야 함.

---

## 5. 권장 우선순위

1. **레퍼런스 영상**: 사람 댄스 대신 **강아지/캐릭터에 가까운 모션** 영상 사용 또는 LTX만으로 모션 전달 품질 확인  
2. **pose_sdxl**: `pipelines/dance/dog_pose_generation.json`을 ComfyUI에서 **실제 SDXL+ControlNet+IPAdapter**로 교체 후 내보내기  
3. **자동 강아지 검출**: 제품 요구 시 `POST /api/...` 단계에 경량 검출기 추가 (별도 이슈)

관련: `pipelines/dance/README.md`, `docs/COMFYUI_FULL_SETUP.md`, `ComfyUI/TROUBLESHOOTING.md` (레포 루트).
