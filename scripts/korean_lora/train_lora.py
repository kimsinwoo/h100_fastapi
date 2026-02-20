#!/usr/bin/env python3
"""
한국어 SFT 데이터로 LoRA 파인튜닝하는 스크립트.
- 데이터: data/korean_sft_train.jsonl (messages 형식)
- 출력: LoRA 어댑터가 저장된 디렉터리 (기본 output/korean_lora)
사용 예:
  python train_lora.py --model_name_or_path openai/gpt-oss-20b --output_dir output/korean_lora
  python train_lora.py --model_name_or_path meta-llama/Llama-3.2-3B --output_dir output/korean_llama_lora
"""
from __future__ import annotations

import os
import sys
from pathlib import Path

from datasets import load_dataset
from peft import LoraConfig, TaskType
from trl import SFTConfig, SFTTrainer

SCRIPT_DIR = Path(__file__).resolve().parent
DATA_DIR = SCRIPT_DIR / "data"
DEFAULT_DATA_FILE = DATA_DIR / "korean_sft_train.jsonl"


def main():
    import argparse
    parser = argparse.ArgumentParser(description="한국어 SFT LoRA 파인튜닝")
    parser.add_argument("--model_name_or_path", type=str, default="Qwen/Qwen2.5-1.5B", help="베이스 모델")
    parser.add_argument("--output_dir", type=str, default=str(SCRIPT_DIR / "output" / "korean_lora"))
    parser.add_argument("--data_file", type=str, default=str(DEFAULT_DATA_FILE))
    parser.add_argument("--num_epochs", type=int, default=3)
    parser.add_argument("--per_device_train_batch_size", type=int, default=2)
    parser.add_argument("--gradient_accumulation_steps", type=int, default=4)
    parser.add_argument("--learning_rate", type=float, default=1e-4)
    parser.add_argument("--max_seq_length", type=int, default=1024)
    parser.add_argument("--lora_r", type=int, default=16)
    parser.add_argument("--lora_alpha", type=int, default=32)
    parser.add_argument("--lora_dropout", type=float, default=0.05)
    parser.add_argument("--hf_token", type=str, default=os.environ.get("HF_TOKEN", ""))
    parser.add_argument("--use_4bit", action="store_true", help="4bit 양자화로 메모리 절약")
    args = parser.parse_args()

    if not Path(args.data_file).exists():
        print(f"데이터 파일이 없습니다: {args.data_file}", file=sys.stderr)
        print("먼저 python data/build_korean_sft.py 를 실행하세요.", file=sys.stderr)
        sys.exit(1)

    token = args.hf_token or os.environ.get("HUGGING_FACE_HUB_TOKEN")
    if token:
        os.environ["HF_TOKEN"] = token

    try:
        import torch
        use_cuda = torch.cuda.is_available()
    except Exception:
        use_cuda = False

    print("데이터셋 로드 중:", args.data_file)
    dataset = load_dataset("json", data_files=args.data_file, split="train")

    lora_config = LoraConfig(
        r=args.lora_r,
        lora_alpha=args.lora_alpha,
        lora_dropout=args.lora_dropout,
        bias="none",
        task_type=TaskType.CAUSAL_LM,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
    )

    training_args = SFTConfig(
        output_dir=args.output_dir,
        num_train_epochs=args.num_epochs,
        per_device_train_batch_size=args.per_device_train_batch_size,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        learning_rate=args.learning_rate,
        max_length=args.max_seq_length,
        dataset_text_field="messages",
        save_strategy="epoch",
        logging_steps=10,
        bf16=use_cuda,
        fp16=not use_cuda,
        gradient_checkpointing=True,
    )

    trainer = SFTTrainer(
        model=args.model_name_or_path,
        args=training_args,
        train_dataset=dataset,
        peft_config=lora_config,
    )
    trainer.train()
    trainer.save_model(args.output_dir)
    print("저장 완료:", args.output_dir)


if __name__ == "__main__":
    main()
