"""
LLM: 로컬 모델(transformers) 로드·추론 또는 OpenAI 호환 API 호출.
동시 사용자 대응: 세마포어로 동시 요청 수 제한.
자연어 처리: 한국어/영어 구분 없이 자연스러운 대화·프롬프트 추천.
"""

from __future__ import annotations

import asyncio
import logging
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


def _load_local_model_sync() -> tuple[Any, Any]:
    """로컬 모델·토크나이저 로드 (한 번만, 동기)."""
    global _local_model, _local_tokenizer
    if _local_model is not None:
        return _local_model, _local_tokenizer
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    settings = get_settings()
    model_id = settings.llm_local_model_id
    device = "cuda" if torch.cuda.is_available() else "cpu"
    dtype = torch.bfloat16 if (device == "cuda" and torch.cuda.is_bf16_supported()) else torch.float16
    logger.info("Loading local LLM: %s on %s", model_id, device)
    tokenizer = AutoTokenizer.from_pretrained(model_id, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        model_id,
        torch_dtype=dtype,
        device_map="auto" if device == "cuda" else None,
        trust_remote_code=True,
    )
    if device == "cpu":
        model = model.to(device)
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
            max_new_tokens=max_tokens,
            do_sample=temperature > 0,
            temperature=temperature if temperature > 0 else 1.0,
            top_p=0.9,
            pad_token_id=tokenizer.eos_token_id,
        )
        out = tokenizer.decode(gen[0][inputs.input_ids.shape[1] :], skip_special_tokens=True)
        return out.strip() or None
    except Exception as e:
        logger.warning("Local LLM inference failed: %s", e)
        return None


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
    return (text.strip() if isinstance(text, str) and text.strip() else None) or None


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


async def complete_chat(messages: list[dict[str, str]], max_tokens: int = 512, temperature: float = 0.7) -> str | None:
    """
    채팅용: 시스템 프롬프트를 넣어 사용자 언어에 맞춰 자연스럽게 응답.
    messages에 system이 없으면 CHAT_SYSTEM_PROMPT를 맨 앞에 삽입.
    """
    if not messages or (messages and (messages[0].get("role") or "").lower() != "system"):
        messages = [{"role": "system", "content": CHAT_SYSTEM_PROMPT}] + list(messages)
    return await complete(messages, max_tokens=max_tokens, temperature=temperature)


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
