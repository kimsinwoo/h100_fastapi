#!/usr/bin/env python3
"""
한국어 SFT LoRA 파인튜닝 (gpt-oss-20b 호환).
- messages: system / user / assistant 구조
- loss는 assistant 부분에만 적용 (system/user는 label -100)
- packing 시 message 단위 유지, EOS 명확 삽입
사용 예:
  python train_lora.py --model_name_or_path openai/gpt-oss-20b --data_file data/korean_sft_train_cleaned.jsonl
"""
from __future__ import annotations

import os
import sys
from pathlib import Path

SCRIPT_DIR = Path(__file__).resolve().parent
DATA_DIR = SCRIPT_DIR / "data"
DEFAULT_DATA_FILE = DATA_DIR / "korean_sft_train_cleaned.jsonl"


def _get_response_template(tokenizer) -> str:
    """채팅 템플릿에서 assistant 응답 시작 문자열 (DataCollatorForCompletionOnlyLM용)."""
    for phrase in ["Assistant:", "assistant\n", "assistant", "### Assistant:"]:
        if tokenizer.encode(phrase, add_special_tokens=False):
            return phrase
    return "assistant"


def main():
    import argparse
    parser = argparse.ArgumentParser(description="한국어 SFT LoRA (assistant-only loss)")
    parser.add_argument("--model_name_or_path", type=str, default="Qwen/Qwen2.5-1.5B")
    parser.add_argument("--output_dir", type=str, default=str(SCRIPT_DIR / "output" / "korean_lora"))
    parser.add_argument("--data_file", type=str, default=str(DEFAULT_DATA_FILE))
    parser.add_argument("--num_epochs", type=int, default=3)
    parser.add_argument("--per_device_train_batch_size", type=int, default=2)
    parser.add_argument("--gradient_accumulation_steps", type=int, default=4)
    parser.add_argument("--learning_rate", type=float, default=1e-4)
    parser.add_argument("--max_seq_length", type=int, default=4096)
    parser.add_argument("--lora_r", type=int, default=32)
    parser.add_argument("--lora_alpha", type=int, default=64)
    parser.add_argument("--lora_dropout", type=float, default=0.05)
    parser.add_argument("--warmup_ratio", type=float, default=0.03)
    parser.add_argument("--label_smoothing", type=float, default=0.05)
    parser.add_argument("--hf_token", type=str, default=os.environ.get("HF_TOKEN", ""))
    parser.add_argument("--use_4bit", action="store_true")
    parser.add_argument("--no_flash_attention", action="store_true", help="flash attention 비활성화")
    args = parser.parse_args()

    if not Path(args.data_file).exists():
        print(f"데이터 파일 없음: {args.data_file}", file=sys.stderr)
        print("먼저 data/build_korean_sft.py 및 data/clean_dataset.py 를 실행하세요.", file=sys.stderr)
        sys.exit(1)

    if args.hf_token:
        os.environ["HF_TOKEN"] = args.hf_token
    token = args.hf_token or os.environ.get("HUGGING_FACE_HUB_TOKEN")
    if token:
        os.environ["HF_TOKEN"] = token

    try:
        import torch
        use_cuda = torch.cuda.is_available()
    except Exception:
        use_cuda = False

    from datasets import load_dataset
    from transformers import AutoModelForCausalLM, AutoTokenizer, TrainingArguments
    from peft import LoraConfig, TaskType
    from trl import SFTConfig, SFTTrainer
    from trl import DataCollatorForCompletionOnlyLM

    print("토크나이저 로드:", args.model_name_or_path)
    tokenizer = AutoTokenizer.from_pretrained(
        args.model_name_or_path,
        trust_remote_code=True,
    )
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token_id = tokenizer.eos_token_id

    response_template = _get_response_template(tokenizer)
    print("response_template (completion-only loss):", repr(response_template))

    data_collator = DataCollatorForCompletionOnlyLM(
        response_template=response_template,
        tokenizer=tokenizer,
        mlm=False,
    )

    print("데이터셋 로드:", args.data_file)
    dataset = load_dataset("json", data_files=args.data_file, split="train")

    lora_config = LoraConfig(
        r=args.lora_r,
        lora_alpha=args.lora_alpha,
        lora_dropout=args.lora_dropout,
        bias="none",
        task_type=TaskType.CAUSAL_LM,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
    )

    # flash attention: 가능하면 사용
    attn_impl = None
    if use_cuda and not args.no_flash_attention:
        try:
            import torch.backends.cuda
            if hasattr(torch.backends.cuda, "flash_sdp_enabled"):
                attn_impl = "flash_attention_2"
        except Exception:
            pass
        if attn_impl is None and hasattr(AutoModelForCausalLM, "from_pretrained"):
            # transformers 4.36+ attn_implementation
            try:
                attn_impl = "flash_attention_2"
            except Exception:
                attn_impl = None

    model_kwargs = {}
    if attn_impl:
        model_kwargs["attn_implementation"] = attn_impl

    training_args = SFTConfig(
        output_dir=args.output_dir,
        num_train_epochs=args.num_epochs,
        per_device_train_batch_size=args.per_device_train_batch_size,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        learning_rate=args.learning_rate,
        max_length=args.max_seq_length,
        warmup_ratio=args.warmup_ratio,
        label_smoothing_factor=args.label_smoothing,
        dataset_text_field="messages",
        packing=False,
        save_strategy="epoch",
        logging_steps=10,
        bf16=use_cuda,
        fp16=not use_cuda and use_cuda is False,
        gradient_checkpointing=True,
        dataloader_pin_memory=use_cuda,
    )

    trainer = SFTTrainer(
        model=args.model_name_or_path,
        model_init_kwargs=model_kwargs if model_kwargs else None,
        args=training_args,
        train_dataset=dataset,
        peft_config=lora_config,
        data_collator=data_collator,
        tokenizer=tokenizer,
    )
    trainer.train()
    trainer.save_model(args.output_dir)
    tokenizer.save_pretrained(args.output_dir)
    print("저장 완료:", args.output_dir)


if __name__ == "__main__":
    main()
