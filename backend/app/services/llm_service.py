"""
LLM: 로컬 모델(transformers) 로드·추론 또는 OpenAI 호환 API 호출.
동시 사용자 대응: 세마포어로 동시 요청 수 제한.
자연어 처리: 한국어/영어 구분 없이 자연스러운 대화·프롬프트 추천.
"""

from __future__ import annotations

import asyncio
import logging
import re
from pathlib import Path
from typing import Any

from app.core.config import get_settings

logger = logging.getLogger(__name__)

_llm_semaphore: asyncio.Semaphore | None = None
_local_model: Any = None
_local_tokenizer: Any = None
_local_lock = asyncio.Lock()


def _get_llm_semaphore() -> asyncio.Semaphore:
    global _llm_semaphore
    if _llm_semaphore is None:
        _llm_semaphore = asyncio.Semaphore(get_settings().llm_max_concurrent)
    return _llm_semaphore


def is_llm_available() -> bool:
    """로컬 LLM 사용 가능 또는 외부 API 베이스 설정 시 True."""
    s = get_settings()
    if not s.llm_enabled:
        return False
    if s.llm_use_local:
        return True
    return bool(s.llm_api_base and s.llm_api_base.strip())


async def preload_local_model() -> None:
    """서버 기동 시 로컬 LLM 미리 로드. llm_use_local일 때만 실행(블로킹이므로 executor에서 실행)."""
    s = get_settings()
    if not s.llm_enabled or not s.llm_use_local:
        return
    loop = asyncio.get_event_loop()
    try:
        await loop.run_in_executor(None, load_local_model_sync)
        logger.info("Local LLM preloaded at startup: %s", s.llm_local_model_id)
    except Exception as e:
        logger.warning("Local LLM preload at startup failed (will load on first request): %s", e)


def _get_hf_token() -> str | None:
    """Hugging Face 토큰 (게이트 모델용). 설정 또는 HF_TOKEN 환경변수."""
    import os
    s = get_settings()
    if s.llm_hf_token and s.llm_hf_token.strip():
        return s.llm_hf_token.strip()
    return os.environ.get("HF_TOKEN") or os.environ.get("HUGGING_FACE_HUB_TOKEN") or None


def load_local_model_sync() -> tuple[Any, Any]:
    """로컬 모델·토크나이저 로드 (한 번만, 동기). 서버 기동 시 또는 첫 요청 시 호출."""
    return _load_local_model_sync()


def _load_local_model_sync() -> tuple[Any, Any]:
    """로컬 모델·토크나이저 로드 (한 번만, 동기). 게이트 모델은 HF 토큰 필요."""
    global _local_model, _local_tokenizer
    if _local_model is not None:
        return _local_model, _local_tokenizer
    import os
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    settings = get_settings()
    model_id = settings.llm_local_model_id
    token = _get_hf_token()
    if token:
        os.environ["HF_TOKEN"] = token  # huggingface_hub가 읽도록
    device = "cuda" if torch.cuda.is_available() else "cpu"
    dtype = torch.bfloat16 if (device == "cuda" and torch.cuda.is_bf16_supported()) else torch.float16
    logger.info("Loading local LLM: %s on %s", model_id, device)
    tokenizer = AutoTokenizer.from_pretrained(model_id, trust_remote_code=True, token=token)
    model = AutoModelForCausalLM.from_pretrained(
        model_id,
        torch_dtype=dtype,
        device_map="auto" if device == "cuda" else None,
        trust_remote_code=True,
        token=token,
    )
    if device == "cpu":
        model = model.to(device)
    # 파인튜닝된 한국어 LoRA가 있으면 적용 (채팅 품질 향상)
    lora_dir = Path(get_settings().korean_lora_output_dir)
    if (lora_dir / "adapter_config.json").exists():
        try:
            from peft import PeftModel
            model = PeftModel.from_pretrained(model, str(lora_dir))
            logger.info("한국어 LoRA 어댑터 로드 완료: %s", lora_dir)
        except Exception as e:
            logger.warning("LoRA 어댑터 로드 실패(베이스 모델만 사용): %s", e)
    _local_tokenizer = tokenizer
    _local_model = model
    return model, tokenizer


def _run_local_inference_sync(
    messages: list[dict[str, str]],
    max_tokens: int = 256,
    temperature: float = 0.7,
) -> str | None:
    """동기 로컬 추론. 메시지를 채팅 형식으로 넣고 생성."""
    try:
        model, tokenizer = _load_local_model_sync()
    except Exception as e:
        logger.exception("Local LLM load failed: %s", e)
        return None

    # apply_chat_template 지원 시 사용, 아니면 단순 포맷
    if hasattr(tokenizer, "apply_chat_template") and tokenizer.chat_template is not None:
        try:
            text = tokenizer.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=True,
            )
        except Exception:
            text = _messages_to_plain(messages)
    else:
        text = _messages_to_plain(messages)

    try:
        inputs = tokenizer(text, return_tensors="pt").to(model.device)
        gen = model.generate(
            **inputs,
            max_new_tokens=min(max_tokens, 2048),
            do_sample=True,
            temperature=0.7,
            top_p=0.9,
            repetition_penalty=1.15,
            no_repeat_ngram_size=4,
            pad_token_id=tokenizer.eos_token_id,
            eos_token_id=tokenizer.eos_token_id,
        )
        out = tokenizer.decode(gen[0][inputs.input_ids.shape[1] :], skip_special_tokens=True)
        out = clean_output(out.strip() or "")
        return out or None
    except Exception as e:
        logger.warning("Local LLM inference failed: %s", e)
        return None


def clean_output(text: str) -> str:
    """
    추론 출력 후처리: 동일 문장 2회 이상 제거, 연속 구두점 정규화,
    따옴표 중복 제거, 빈 줄 3줄 이상 제거, JSON 구조·비정상 문자 제거.
    """
    if not text or not text.strip():
        return text
    s = text.strip()
    # 연속 구두점 2회 이상 → 1회
    s = re.sub(r"([.,;:!?\-~])\s*\1+", r"\1", s)
    # 공백 3회 이상 → 1회
    s = re.sub(r" {3,}", " ", s)
    # 따옴표 중복 (" "" " 등) → 하나로
    s = re.sub(r'"\s*"+', '"', s)
    # 빈 줄 3줄 이상 → 2줄
    s = re.sub(r"\n{3,}", "\n\n", s)
    # 동일 문장 2회 이상 제거 (줄 단위)
    lines = s.split("\n")
    seen: set[str] = set()
    out = []
    for line in lines:
        key = line.strip()[:80] if line.strip() else ""
        if key and key in seen:
            continue
        if key:
            seen.add(key)
        out.append(line)
    s = "\n".join(out)
    # JSON 조각 제거 (키:값 블록)
    s = re.sub(r'\s*\{\s*"[^"]+"\s*:\s*[^}]*\}\s*', " ", s)
    s = re.sub(r"\s*\[\s*[^\]]*\]\s*", " ", s)
    # 비정상 문자 제거
    s = "".join(c for c in s if c in "\n\t" or (ord(c) >= 32 and ord(c) != 127))
    return s.strip()


def _messages_to_plain(messages: list[dict[str, str]]) -> str:
    """메시지 리스트를 단순 'User/Assistant' 텍스트로 변환."""
    parts = []
    for m in messages:
        role = (m.get("role") or "user").lower()
        content = (m.get("content") or "").strip()
        if not content:
            continue
        if role == "system":
            parts.append(f"System: {content}\n")
        elif role == "user":
            parts.append(f"User: {content}\n")
        elif role == "assistant":
            parts.append(f"Assistant: {content}\n")
    parts.append("Assistant: ")
    return "".join(parts)


async def _complete_via_api(
    messages: list[dict[str, str]],
    max_tokens: int,
    temperature: float,
) -> str | None:
    """외부 OpenAI 호환 API 호출."""
    import httpx

    settings = get_settings()
    url = f"{settings.llm_api_base.rstrip('/')}/chat/completions"
    payload: dict[str, Any] = {
        "model": settings.llm_model,
        "messages": messages,
        "max_tokens": max_tokens,
        "temperature": temperature,
        "stream": False,
    }
    headers: dict[str, str] = {"Content-Type": "application/json"}
    if settings.llm_api_key and settings.llm_api_key.strip():
        headers["Authorization"] = f"Bearer {settings.llm_api_key.strip()}"

    async with httpx.AsyncClient(timeout=settings.llm_timeout_seconds) as client:
        r = await client.post(url, json=payload, headers=headers)
        r.raise_for_status()
        data = r.json()
    choices = data.get("choices") or []
    if not choices:
        return None
    content = choices[0].get("message") or {}
    text = content.get("content")
    raw = (text.strip() if isinstance(text, str) and text.strip() else None) or None
    return clean_output(raw) if raw else None


async def complete(
    messages: list[dict[str, str]],
    *,
    max_tokens: int = 256,
    temperature: float = 0.7,
) -> str | None:
    """
    채팅 완성: 로컬 모델 또는 외부 API.
    사용 언어(한국어/영어)에 맞춰 자연스럽게 응답하도록 시스템 프롬프트에서 유도.
    """
    if not is_llm_available():
        return None
    settings = get_settings()
    sem = _get_llm_semaphore()

    async with sem:
        if settings.llm_use_local:
            loop = asyncio.get_event_loop()
            return await loop.run_in_executor(
                None,
                lambda: _run_local_inference_sync(messages, max_tokens, temperature),
            )
        try:
            return await _complete_via_api(messages, max_tokens, temperature)
        except Exception as e:
            logger.warning("LLM API request failed: %s", e)
            return None


# ---------- 자연어 처리: 자연스러운 채팅·프롬프트 추천 ----------

CHAT_SYSTEM_PROMPT = (
    "You are a helpful, friendly assistant. "
    "Respond naturally and conversationally in the same language the user uses (e.g. Korean or English). "
    "Keep responses clear and concise unless the user asks for more detail. "
    "Be polite and avoid redundant phrases."
)

PROMPT_SUGGEST_SYSTEM = (
    "You are a prompt writer for an image-to-image AI. "
    "The user may give hints in Korean or English. "
    "Output exactly one short English phrase suitable as an image generation prompt (comma-separated keywords OK). "
    "No explanation, no quotes, no extra text."
)

# 건강 질문 도우미 AI — 한 번만 생성, 한글만 출력
HEALTH_DISCLAIMER = "정확한 판단은 의료·수의 전문가에게 확인하세요."

# 후처리에서 제거할 문구: 시스템 프롬프트/지시가 그대로 출력된 줄
_HEALTH_INSTRUCTION_MARKERS = (
    "알파벳",
    "로마자",
    "한 글자도 쓰지 않는다",
    "접두어",
    "소제목",
)

HEALTH_ASSISTANT_SYSTEM_PROMPT = """당신은 건강 질문 도우미입니다. 사용자 증상·반려동물 상황에 대해 참고 정보만 제공합니다.

[최우선] 답변은 한글만 사용한다. 알파벳·영문·로마자 한 글자도 쓰지 않는다. 고유명사·기술 용어도 한글로 쓴다. 접두어(analysis 등) 금지. 답변은 바로 본문으로 시작한다.

[규칙] 진단·확정 표현 금지. "참고로 생각해볼 수 있는 내용", "병원에서 확인해 보시면 좋습니다"처럼 안내만. 소제목은 **굵게**. 목록은 한 줄에 "• 항목". 문장 끝까지 완성. 반드시 "정확한 판단은 의료·수의 전문가에게 확인하세요" 포함. 반려동물 건강 질문이 아니면 답변하지 않는다."""


async def complete_chat(messages: list[dict[str, str]], max_tokens: int = 512, temperature: float = 0.7) -> str | None:
    """
    채팅용: 시스템 프롬프트를 넣어 사용자 언어에 맞춰 자연스럽게 응답.
    messages에 system이 없으면 CHAT_SYSTEM_PROMPT를 맨 앞에 삽입.
    """
    if not messages or (messages and (messages[0].get("role") or "").lower() != "system"):
        messages = [{"role": "system", "content": CHAT_SYSTEM_PROMPT}] + list(messages)
    return await complete(messages, max_tokens=max_tokens, temperature=temperature)


def _strip_alphabet(text: str) -> str:
    """알파벳(a-zA-Z)만 제거. 한글·숫자·•·**·공백·줄바꿈 등은 유지."""
    if not text or not text.strip():
        return text
    cleaned = re.sub(r"[a-zA-Z]", "", text)
    cleaned = re.sub(r"\n{3,}", "\n\n", cleaned).strip()
    return _strip_leading_junk(cleaned)


def _strip_leading_junk(text: str) -> str:
    """알파벳 제거 후 남은 앞쪽 구두점·쓰레기 줄 제거. 본문(한글/•/**)만 남김."""
    if not text or not text.strip():
        return text
    lines = text.split("\n")
    filtered = []
    for line in lines:
        s = line.strip()
        if not s:
            if filtered:
                filtered.append(line)
            continue
        # 구두점·공백만 있는 줄 제거
        if not re.search(r"[가-힣]", s) and "•" not in s and not s.startswith("**"):
            continue
        # 사용자 말 반복처럼 보이는 짧은 인용 한 줄 제거 (전체가 따옴표 한 덩어리)
        if re.match(r'^\s*"[^"]{1,80}"\s*$', line.strip()):
            continue
        filtered.append(line)
    if not filtered:
        return text.strip()
    merged = "\n".join(filtered)
    # 앞쪽에 disclaimer + 구두점만 있고 그 다음에 ** 제목이 오면, ** 부터 시작하도록 자름
    idx = merged.find("**")
    if idx > 0 and idx < 120:
        before = re.sub(r"[^\uac00-\ud7a3·]", "", merged[:idx])
        if "정확한판단은의료수의전문가에게확인하세요" in before or len(before) < 30:
            merged = merged[idx:].strip()
    # 맨 앞 구두점·공백만 제거(첫 한글/•/** 나올 때까지)
    for i, c in enumerate(merged):
        if "\uac00" <= c <= "\ud7a3" or c == "•":
            return merged[i:].strip()
        if i + 2 <= len(merged) and merged[i : i + 2] == "**":
            return merged[i:].strip()
    return merged.strip()


def _is_instruction_leakage(s: str) -> bool:
    """시스템 프롬프트/지시가 그대로 출력된 줄이면 True. 이런 줄은 제거 대상."""
    for m in _HEALTH_INSTRUCTION_MARKERS:
        if m in s:
            return True
    # 줄 내용이 "수의", "수의사", "의료" 반복만 있는 짧은 경우 (지시 유출)
    only_korean = re.sub(r"[^\uac00-\ud7a3]", "", s)
    if len(only_korean) <= 24 and re.fullmatch(r"(수의|수의사|의료)+", only_korean):
        return True
    return False


def _dedupe_and_fix_disclaimer(text: str) -> str:
    """반복 문구·disclaimer·지시 유출 제거. disclaimer는 맨 끝에 한 번만."""
    if not text or not text.strip():
        return text
    raw = text.strip()
    pattern = re.escape(HEALTH_DISCLAIMER.rstrip("."))
    disclaimer_only = re.compile(r"^\s*" + pattern + r"\.?\s*$")
    lines = raw.split("\n")
    kept = []
    prev = None
    hangul = re.compile(r"[가-힣]")
    for line in lines:
        s = line.strip()
        if not s:
            if kept:
                kept.append(line)
            continue
        if disclaimer_only.match(s):
            continue
        # disclaimer 문구만 있고 구두점만 더 있는 줄 제거
        normalized = re.sub(r"[^\uac00-\ud7a3·]", "", s)
        if normalized == "정확한판단은의료수의전문가에게확인하세요":
            continue
        if _is_instruction_leakage(s):
            continue
        # 한글 거의 없고 구두점·따옴표만 많은 줄 제거 (한글 2글자 미만)
        if len(hangul.findall(s)) < 2 and not s.startswith("**") and "•" not in s:
            continue
        # 같은 줄 연속 반복 제거
        if s == prev:
            continue
        prev = s
        kept.append(line)
    if not kept:
        return HEALTH_DISCLAIMER
    merged = "\n".join(kept).strip()
    base = HEALTH_DISCLAIMER.rstrip(".")
    if merged.endswith(HEALTH_DISCLAIMER) or merged.rstrip().endswith(base):
        return merged
    return merged.rstrip() + "\n\n" + HEALTH_DISCLAIMER


def _last_user_content_has_hangul(messages: list[dict[str, str]]) -> bool:
    """마지막 user 메시지에 한글이 있으면 True."""
    for m in reversed(messages):
        if (m.get("role") or "").lower() == "user":
            return bool(re.search(r"[가-힣]", (m.get("content") or "")))
    return False


async def complete_health_chat(
    messages: list[dict[str, str]],
    max_tokens: int = 1024,
    temperature: float = 0.4,
) -> str | None:
    """건강 질문 도우미. 한 번만 생성. 한국어 대화면 응답에서 알파벳만 후처리로 제거."""
    if not messages or (messages and (messages[0].get("role") or "").lower() != "system"):
        msgs = [{"role": "system", "content": HEALTH_ASSISTANT_SYSTEM_PROMPT}] + list(messages)
    else:
        msgs = [{"role": "system", "content": HEALTH_ASSISTANT_SYSTEM_PROMPT}] + [
            m for m in messages if (m.get("role") or "").lower() != "system"
        ]
    result = await complete(msgs, max_tokens=max_tokens, temperature=temperature)
    if not result:
        return result
    if _last_user_content_has_hangul(msgs):
        result = _strip_alphabet(result)
        if result:
            result = _dedupe_and_fix_disclaimer(result)
    return result


async def suggest_prompt_for_style(style_key: str, user_hint: str | None = None) -> str | None:
    """
    스타일 + 사용자 힌트(한국어/영어 무관)로 이미지 생성용 영어 프롬프트 한 문장 추천.
    """
    from app.models.style_presets import get_style_prompt

    style_desc = get_style_prompt(style_key)
    user_text = f" 사용자 힌트: {user_hint.strip()}" if (user_hint and user_hint.strip()) else ""
    user_msg = (
        f"Style: {style_desc}.{user_text} "
        "Generate one short image prompt in English only (one phrase, comma-separated keywords OK)."
    )
    out = await complete(
        [{"role": "system", "content": PROMPT_SUGGEST_SYSTEM}, {"role": "user", "content": user_msg}],
        max_tokens=120,
        temperature=0.5,
    )
    return out


def get_llm_model_display_name() -> str:
    """프론트 표시용: 로컬이면 모델 ID, 아니면 llm_model."""
    s = get_settings()
    if s.llm_use_local:
        return s.llm_local_model_id
    return s.llm_model
