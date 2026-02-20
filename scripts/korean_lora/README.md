# 한국어 LoRA SFT 파인튜닝

한국어 이해·생성 능력을 높이기 위한 고품질 SFT 데이터와 LoRA 파인튜닝 스크립트입니다.

## 구성

- **data/build_korean_sft.py**  
  학습용 JSONL 생성. 12개 카테고리(일상 대화, IT 기술, 맞춤법, 문체 변환, 사투리, 금융, 법률, 자동차 정비, 감정 상담, 긴 글 요약, 추론, 코드 설명)를 포함해 **100개 이상** 대화형 예시를 만듭니다.
- **data/korean_sft_train.jsonl**  
  위 스크립트 실행 후 생성되는 학습 데이터. 각 줄은 `{"messages": [{"role": "system"|"user"|"assistant", "content": "..."}, ...]}` 형식입니다.
- **train_lora.py**  
  해당 JSONL로 LoRA SFT 학습을 수행합니다. TRL `SFTTrainer` + PEFT LoRA 사용.

## 사용 방법

### 1. 학습 데이터 생성

```bash
cd zimage_webapp/scripts/korean_lora
python data/build_korean_sft.py
```

`data/korean_sft_train.jsonl` 파일이 생성됩니다.

### 2. 학습용 패키지 설치

```bash
pip install -r requirements-train.txt
```

(이미 `zimage_webapp/backend/requirements.txt`를 설치했다면, 여기서 추가로 필요한 것은 `peft`, `trl`, `datasets` 등입니다.)

### 3. LoRA 학습 실행

```bash
# 기본: Qwen2.5-1.5B, output/korean_lora 에 저장
python train_lora.py

# 모델·출력 경로 지정
python train_lora.py --model_name_or_path Qwen/Qwen2.5-1.5B --output_dir output/korean_lora

# 4bit 양자화로 GPU 메모리 절약 (작은 GPU용)
python train_lora.py --use_4bit --per_device_train_batch_size 1
```

Hugging Face 비공개 모델을 쓸 경우:

```bash
export HF_TOKEN=your_token
python train_lora.py --model_name_or_path your-org/your-model
```

### 4. 학습된 LoRA 사용

학습이 끝나면 `output_dir`에 LoRA 어댑터와 토크나이저가 저장됩니다.  
추론 시 베이스 모델과 함께 로드해서 사용하면 됩니다.

```python
from peft import AutoPeftModelForCausalLM
from transformers import AutoTokenizer

model = AutoPeftModelForCausalLM.from_pretrained("output/korean_lora")
tokenizer = AutoTokenizer.from_pretrained("output/korean_lora")
# 이후 generate 등으로 사용
```

## 데이터 조건 (요약)

- 모든 응답은 **자연스러운 한국어**, 번역투 금지.
- 문맥에 맞는 **높임법** 유지.
- **한국어 외 언어 사용 금지**.
- 12개 카테고리 모두 포함, **최소 100개 이상**, **중복 없음**, 다양한 문장 구조.

## 응답 시작 패턴 붕괴 후 재학습 (근본 해결)

BOS 이후 토큰 분포가 깨져 구두점/경고문이 먼저 나올 때, 데이터 리밸런스 + 재미세조정으로 해결합니다.

1. **경고문 비율 5% 이하**: `data/rebalance_warning_ratio.py` 로 기존 정제 데이터에서 "정확한 판단은" 등 포함 샘플을 전체의 5% 이하로 축소.
2. **경고문 없는 정상 데이터 3000건**: `data/build_no_warning_sft.py` 로 경고문 없이 한글 문장으로만 시작하는 SFT 추가.
3. **노이즈→정상 교정 데이터**: `data/build_correction_sft.py` 로 잘못된 시작 패턴 → 올바른 문장 학습.
4. **5e-5, 1 epoch 재미세조정**: 위 데이터를 병합한 뒤 `train_lora.py --learning_rate 5e-5 --num_epochs 1` 로 학습.

한 번에 실행:

```bash
cd zimage_webapp/scripts/korean_lora
chmod +x run_relearn.sh
./run_relearn.sh
```

### H100 서버에서 실행

H100에서는 Flash Attention·bf16·배치 크기를 키워서 더 빠르게 돌릴 수 있습니다.

**1) 한 번만 설정 (권장)**

```bash
pip install flash-attn --no-build-isolation   # H100에서 추론·학습 속도 향상
```

설치가 어려우면 생략해도 됩니다. 이 경우 스크립트가 SDPA로 학습합니다.

**2) 재학습 파이프라인 실행**

```bash
cd zimage_webapp/scripts/korean_lora
chmod +x run_relearn_h100.sh
./run_relearn_h100.sh
```

- `run_relearn_h100.sh`: 데이터 단계는 동일, 학습만 **per_device_train_batch_size=4**, **gradient_accumulation_steps=2**, Flash Attention 사용 시도(기본), 4bit 미사용.
- GPU 메모리 부족 시 `run_relearn.sh`를 쓰거나, `train_lora.py`에 `--per_device_train_batch_size 2 --use_4bit` 등을 넘겨 조정하세요.

수동 순서:

```bash
python data/clean_dataset.py data/korean_sft_train.jsonl -o data/korean_sft_train_cleaned.jsonl
python data/rebalance_warning_ratio.py data/korean_sft_train_cleaned.jsonl -o data/korean_sft_rebalanced.jsonl --max-ratio 0.05
python data/build_no_warning_sft.py -n 3000 -o data/korean_sft_no_warning.jsonl
python data/build_correction_sft.py -n 3500 -o data/korean_sft_correction.jsonl
python data/merge_for_relearn.py -o data/korean_sft_final.jsonl
python train_lora.py --data_file data/korean_sft_final.jsonl --learning_rate 5e-5 --num_epochs 1
```

## 주의

- GPU 메모리가 부족하면 `--use_4bit`, `--per_device_train_batch_size 1`, `--gradient_accumulation_steps 8` 등으로 조정하세요.
- 베이스 모델에 따라 `train_lora.py`의 `target_modules`(예: `q_proj`, `k_proj` 등)를 해당 모델 구조에 맞게 바꿔야 할 수 있습니다.
