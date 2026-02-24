# Z-Image-Turbo LoRA (HF 전용)

- **추가 데이터 수집·외부 결합·로컬 export·synthetic 금지.** 지정 HF 데이터셋만 사용.
- `datasets.load_dataset(..., streaming=False)` 직접 로드, image/caption 컬럼 자동 감지.
- caption 없으면 스타일 토큰만 사용: `3d_render style`, `cyberpunk style`, `pixel art style`.
- 데이터 처리: **resize + center crop만** (color jitter, random flip 등 augmentation 금지).

## 데이터셋 (merge 금지, 스타일당 1개)

| 스타일      | 데이터셋 |
|------------|----------|
| 3d_render  | Guizmus/3DChanStyle |
| cyberpunk  | imadjinn/cyberpunk_diffusion |
| pixel_art  | nightaway/PixelArt2 |

## 학습

```bash
cd zimage_webapp/backend/scripts/zit_lora
python train_lora_zit.py --style 3d_render
python train_lora_zit.py --style cyberpunk
python train_lora_zit.py --style pixel_art
```

- rank: 3D=16, Cyberpunk=16, Pixel=8 / epoch: 6 (pixel=8) / lr=1e-4, adamw_8bit, bfloat16, torch.compile
- 출력: `backend/lora_output/{style}.safetensors`

## 추론 테스트

```bash
python inference_test.py --style pixel_art --output out.png
```

- `--style`으로 해당 LoRA 자동 로드 (`lora_output/{style}.safetensors`).
