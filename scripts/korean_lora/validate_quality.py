#!/usr/bin/env python3
"""
학습 완료 후 자동 품질 검증: 랜덤 프롬프트 50개로 생성 후
반복률·구두점 이상·중복 문장·따옴표 노이즈·문장 완성도 계산, 리포트 출력.
"""
from __future__ import annotations

import re
import sys
from pathlib import Path

SCRIPT_DIR = Path(__file__).resolve().parent


def _repetition_rate(text: str) -> float:
    """동일 문장 2회 이상 반복 비율 (문장 수 대비 중복 제거된 문장 수)."""
    if not text or not text.strip():
        return 0.0
    sentences = re.split(r"[.!?]\s+", text)
    sentences = [s.strip() for s in sentences if len(s.strip()) >= 5]
    if not sentences:
        return 0.0
    seen = set()
    duplicates = 0
    for s in sentences:
        key = s[:80]
        if key in seen:
            duplicates += 1
        else:
            seen.add(key)
    return (duplicates / len(sentences)) * 100.0 if sentences else 0.0


def _punctuation_anomaly_ratio(text: str) -> float:
    """구두점 비정상 밀도: 연속 구두점 2회 이상인 위치 비율."""
    if not text or not text.strip():
        return 0.0
    total = len(text)
    anomaly_count = len(re.findall(r"[.,;:!?\-~]\s*[.,;:!?\-~]", text))
    return (anomaly_count / total) * 100.0 if total else 0.0


def _duplicate_sentence_ratio(text: str) -> float:
    """중복 문장 비율 (전체 문장 중 두 번째 이상 출현 비율)."""
    return _repetition_rate(text)


def _quote_noise_ratio(text: str) -> float:
    """따옴표 노이즈: 연속 따옴표/불필요 따옴표 비율. 0% 목표."""
    if not text or not text.strip():
        return 0.0
    # 연속 따옴표 패턴
    noise = len(re.findall(r'"\s*"+', text)) + len(re.findall(r'\s*"\s*"\s*', text))
    return (noise / max(len(text), 1)) * 100.0


def _sentence_completeness(text: str) -> float:
    """문장 완성도: 마침표/물음표/느낌표로 끝나는 문장 비율 (95% 이상 목표)."""
    if not text or not text.strip():
        return 100.0
    sentences = re.split(r"(?<=[.!?])\s+", text)
    sentences = [s for s in sentences if s.strip()]
    if not sentences:
        return 0.0
    completed = sum(1 for s in sentences if s.strip() and s.rstrip()[-1] in ".!?")
    return (completed / len(sentences)) * 100.0 if sentences else 0.0


def run_validation(
    model_path: str,
    lora_path: str | None,
    data_file: str | None,
    num_samples: int = 50,
    max_new_tokens: int = 256,
) -> dict:
    """모델 로드 후 num_samples개 프롬프트로 생성해 지표 계산."""
    import json
    import random

    try:
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer
    except ImportError:
        return {"error": "transformers/torch not installed"}

    if data_file and Path(data_file).exists():
        with open(data_file, "r", encoding="utf-8") as f:
            lines = [l for l in f if l.strip()]
        if lines:
            samples = []
            for _ in range(min(num_samples, len(lines) * 2)):
                line = random.choice(lines)
                try:
                    item = json.loads(line)
                    msgs = item.get("messages") or []
                    for m in msgs:
                        if (m.get("role") or "").lower() == "user":
                            samples.append((m.get("content") or "").strip())
                            break
                except Exception:
                    pass
            prompts = [p for p in samples if p][:num_samples]
        else:
            prompts = []
    else:
        prompts = []

    if len(prompts) < num_samples:
        # 기본 프롬프트로 보충
        defaults = [
            "오늘 기분이 어때?",
            "강아지 키우고 있어?",
            "점심 뭐 먹을까?",
            "날씨가 좋네.",
            "요즘 잠 잘 자?",
        ]
        while len(prompts) < num_samples:
            prompts.append(random.choice(defaults))

    print("모델 로드 중:", model_path, file=sys.stderr)
    tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token_id = tokenizer.eos_token_id
    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        torch_dtype=torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16,
        device_map="auto" if torch.cuda.is_available() else None,
        trust_remote_code=True,
    )
    if lora_path and Path(lora_path).exists() and (Path(lora_path) / "adapter_config.json").exists():
        from peft import PeftModel
        model = PeftModel.from_pretrained(model, lora_path)
        print("LoRA 로드:", lora_path, file=sys.stderr)

    repetition_penalty = 1.15
    no_repeat_ngram_size = 4
    results = []
    for i, prompt in enumerate(prompts[:num_samples]):
        messages = [{"role": "user", "content": prompt}]
        if hasattr(tokenizer, "apply_chat_template") and tokenizer.chat_template:
            text = tokenizer.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True
            )
        else:
            text = f"User: {prompt}\nAssistant: "
        inputs = tokenizer(text, return_tensors="pt").to(model.device)
        with torch.no_grad():
            gen = model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                do_sample=True,
                temperature=0.7,
                top_p=0.9,
                repetition_penalty=repetition_penalty,
                no_repeat_ngram_size=no_repeat_ngram_size,
                pad_token_id=tokenizer.eos_token_id,
                eos_token_id=tokenizer.eos_token_id,
            )
        out = tokenizer.decode(gen[0][inputs.input_ids.shape[1] :], skip_special_tokens=True)
        out = out.strip()
        results.append({
            "prompt": prompt[:50],
            "output": out,
            "repetition_rate": _repetition_rate(out),
            "punctuation_anomaly": _punctuation_anomaly_ratio(out),
            "duplicate_sentence": _duplicate_sentence_ratio(out),
            "quote_noise": _quote_noise_ratio(out),
            "completeness": _sentence_completeness(out),
        })

    avg_rep = sum(r["repetition_rate"] for r in results) / len(results)
    avg_punct = sum(r["punctuation_anomaly"] for r in results) / len(results)
    avg_dup = sum(r["duplicate_sentence"] for r in results) / len(results)
    avg_quote = sum(r["quote_noise"] for r in results) / len(results)
    avg_comp = sum(r["completeness"] for r in results) / len(results)

    # 기준: 반복률 1% 이하, 구두점 이상 2% 이하, 따옴표 노이즈 0%, 완성도 95% 이상
    passed_rep = avg_rep <= 1.0
    passed_punct = avg_punct <= 2.0
    passed_quote = avg_quote <= 0.01
    passed_comp = avg_comp >= 95.0
    all_passed = passed_rep and passed_punct and passed_quote and passed_comp

    report = {
        "num_samples": len(results),
        "repetition_rate_avg_pct": round(avg_rep, 2),
        "punctuation_anomaly_avg_pct": round(avg_punct, 2),
        "duplicate_sentence_avg_pct": round(avg_dup, 2),
        "quote_noise_avg_pct": round(avg_quote, 2),
        "sentence_completeness_avg_pct": round(avg_comp, 2),
        "criteria": {
            "repetition_rate_le_1_pct": passed_rep,
            "punctuation_anomaly_le_2_pct": passed_punct,
            "quote_noise_zero": passed_quote,
            "completeness_ge_95_pct": passed_comp,
        },
        "all_passed": all_passed,
        "samples": results[:5],
    }
    return report


def main() -> None:
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="Qwen/Qwen2.5-1.5B")
    ap.add_argument("--lora", default=None, help="LoRA adapter path")
    ap.add_argument("--data_file", default=None, help="JSONL for random prompts")
    ap.add_argument("--num_samples", type=int, default=50)
    ap.add_argument("--max_new_tokens", type=int, default=256)
    args = ap.parse_args()
    if not args.lora:
        args.lora = str(SCRIPT_DIR / "output" / "korean_lora")
    report = run_validation(
        args.model,
        args.lora,
        args.data_file,
        num_samples=args.num_samples,
        max_new_tokens=args.max_new_tokens,
    )
    if report.get("error"):
        print(report["error"], file=sys.stderr)
        sys.exit(1)
    print("========== 품질 검증 리포트 ==========")
    print(f"샘플 수: {report['num_samples']}")
    print(f"동일 문장 반복률 평균: {report['repetition_rate_avg_pct']}% (기준 ≤1%) {'PASS' if report['criteria']['repetition_rate_le_1_pct'] else 'FAIL'}")
    print(f"구두점 이상 비율 평균: {report['punctuation_anomaly_avg_pct']}% (기준 ≤2%) {'PASS' if report['criteria']['punctuation_anomaly_le_2_pct'] else 'FAIL'}")
    print(f"따옴표 노이즈: {report['quote_noise_avg_pct']}% (기준 0%) {'PASS' if report['criteria']['quote_noise_zero'] else 'FAIL'}")
    print(f"문장 완성도 평균: {report['sentence_completeness_avg_pct']}% (기준 ≥95%) {'PASS' if report['criteria']['completeness_ge_95_pct'] else 'FAIL'}")
    print(f"전체 통과: {'예' if report['all_passed'] else '아니오'}")
    print("======================================")


if __name__ == "__main__":
    main()
