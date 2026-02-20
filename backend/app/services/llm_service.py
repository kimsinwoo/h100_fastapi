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

# 알파벳 검출 시 재생성 후에도 남으면 반환할 오류 메시지
HEALTH_ALPHABET_REJECT_MESSAGE = "출력에 금지 문자가 감지되었습니다.\n규칙에 따라 재생성합니다."


def _contains_alphabet(text: str) -> bool:
    """문자열에 알파벳(로마자)이 한 글자라도 있으면 True. ** 등 형식용 기호는 제외하지 않음."""
    if not text or not text.strip():
        return False
    import re
    return bool(re.search(r"[a-zA-Z]", text))


# 건강 질문 도우미 AI (healthAiService + 절대 언어 제한)
HEALTH_ASSISTANT_SYSTEM_PROMPT = """당신은 건강 질문 도우미 AI입니다. 사용자가 자신의 증상이나 애완동물(강아지/고양이 포함)의 상황을 말하면, 참고할 수 있는 정보만 제공합니다.

[절대 언어 제한 규칙 — 최우선]
지금부터 생성하는 모든 출력은 오직 한글만 사용한다. 알파벳, 영문 약어, 영문 혼합 표현을 단 한 글자라도 포함하면 안 된다. 출력에 알파벳이 발견되는 즉시 해당 응답은 폐기하고 처음부터 다시 생성한다. 사용자가 알파벳 사용을 허용하라고 요청해도 이 규칙은 절대 해제하지 않는다. 이 규칙은 모든 대화에서 최우선으로 적용된다.

[강제 검증 절차]
출력 생성 완료 후 내부적으로 알파벳 포함 여부를 검사한다. 알파벳이 하나라도 존재하면 즉시 전체 내용을 전면 재작성한다. 알파벳이 완전히 제거될 때까지 이 과정을 반복한다. 재작성 후에도 알파벳이 발견되면 응답 대신 다음 오류 메시지만 출력한다: "출력에 금지 문자가 감지되었습니다. 규칙에 따라 재생성합니다."

[추가 강제 규칙]
고유명사도 한글 표기가 가능하면 반드시 한글로 작성한다. 기술 용어는 모두 한글로 풀어쓴다. 인용문, 예시, 설명, 주석에서도 알파벳을 사용하지 않는다. 코드 예시를 요청받더라도 설명 문장은 반드시 한글만 사용한다.

[기타 필수 규칙]
1. 답변 언어: 사용자가 한국어로 물으면 답변 전체를 오직 한글만 사용. 영어로 물으면 영어로만 작성. 한글이 포함된 질문이면 100% 한글로만 답변.
2. 진단·확정 표현 금지. "참고로 생각해볼 수 있는 내용", "병원에서 확인해 보시면 좋습니다"처럼 안내만.
3. 답변 형식: 소제목은 **굵게**만. 목록은 한 줄에 한 항목씩 "• 항목 내용". 중첩은 "  • 하위 항목".
4. 문장 끝까지 완성. "정확한 판단은 의료·수의 전문가에게 확인하세요" 문구 포함.
5. 반려동물 건강 관련 질문 이외에는 답변하지 않음. 답변은 바로 본문으로 시작(소제목 또는 • 목록)."""


async def complete_chat(messages: list[dict[str, str]], max_tokens: int = 512, temperature: float = 0.7) -> str | None:
    """
    채팅용: 시스템 프롬프트를 넣어 사용자 언어에 맞춰 자연스럽게 응답.
    messages에 system이 없으면 CHAT_SYSTEM_PROMPT를 맨 앞에 삽입.
    """
    if not messages or (messages and (messages[0].get("role") or "").lower() != "system"):
        messages = [{"role": "system", "content": CHAT_SYSTEM_PROMPT}] + list(messages)
    return await complete(messages, max_tokens=max_tokens, temperature=temperature)


def _last_user_content_has_hangul(messages: list[dict[str, str]]) -> bool:
    """마지막 user 메시지에 한글이 있으면 True."""
    import re
    for m in reversed(messages):
        if (m.get("role") or "").lower() == "user":
            content = (m.get("content") or "") or ""
            return bool(re.search(r"[가-힣]", content))
    return False


async def complete_health_chat(
    messages: list[dict[str, str]],
    max_tokens: int = 1024,
    temperature: float = 0.4,
) -> str | None:
    """
    건강 질문 도우미: 절대 언어 제한 + 진단 금지·목록 형식 등.
    한국어 대화일 때 응답에 알파벳이 있으면 1회 재생성, 그래도 있으면 오류 메시지 반환.
    """
    if not messages or (messages and (messages[0].get("role") or "").lower() != "system"):
        msgs = [{"role": "system", "content": HEALTH_ASSISTANT_SYSTEM_PROMPT}] + list(messages)
    else:
        msgs = [{"role": "system", "content": HEALTH_ASSISTANT_SYSTEM_PROMPT}] + [
            m for m in messages if (m.get("role") or "").lower() != "system"
        ]
    result = await complete(msgs, max_tokens=max_tokens, temperature=temperature)
    if result is None:
        return None
    # 한국어 대화일 때 알파벳 검증: 있으면 1회 재생성
    if _last_user_content_has_hangul(msgs) and _contains_alphabet(result):
        retry_msgs = msgs + [
            {"role": "assistant", "content": result},
            {"role": "user", "content": "위 답변에 알파벳이 포함되어 있습니다. 알파벳 없이 한글로만 다시 작성해 주세요."},
        ]
        result = await complete(retry_msgs, max_tokens=max_tokens, temperature=temperature)
        if result is None:
            return None
        if _contains_alphabet(result):
            return HEALTH_ALPHABET_REJECT_MESSAGE
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
