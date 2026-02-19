"""
LLM API 연동 (OpenAI 호환). API 키/베이스 설정 시 프롬프트 추천 등에 사용.
gpt-oss-20b 등 오픈소스는 동일 API 스펙으로 베이스 URL/모델만 변경하면 연동 가능.
"""

from __future__ import annotations

import logging
from typing import Any

from app.core.config import get_settings

logger = logging.getLogger(__name__)


def is_llm_available() -> bool:
    """LLM API 사용 가능 여부 (설정 + 키/베이스 있음)."""
    s = get_settings()
    if not s.llm_enabled or not s.llm_api_base or not s.llm_api_key:
        return False
    return True


async def complete(
    messages: list[dict[str, str]],
    *,
    max_tokens: int = 256,
    temperature: float = 0.7,
) -> str | None:
    """
    OpenAI 호환 chat/completions 호출. 성공 시 assistant 메시지 텍스트 반환, 실패 시 None.
    """
    if not is_llm_available():
        return None
    import httpx
    settings = get_settings()
    url = f"{settings.llm_api_base.rstrip('/')}/chat/completions"
    payload: dict[str, Any] = {
        "model": settings.llm_model,
        "messages": messages,
        "max_tokens": max_tokens,
        "temperature": temperature,
    }
    headers = {
        "Authorization": f"Bearer {settings.llm_api_key}",
        "Content-Type": "application/json",
    }
    try:
        async with httpx.AsyncClient(timeout=settings.llm_timeout_seconds) as client:
            r = await client.post(url, json=payload, headers=headers)
            r.raise_for_status()
            data = r.json()
    except Exception as e:
        logger.warning("LLM API request failed: %s", e)
        return None
    choices = data.get("choices") or []
    if not choices:
        return None
    content = choices[0].get("message") or {}
    text = content.get("content")
    return (text.strip() if isinstance(text, str) and text.strip() else None) or None


async def suggest_prompt_for_style(style_key: str, user_hint: str | None = None) -> str | None:
    """
    스타일 + 사용자 힌트를 주면 이미지 생성용 영어 프롬프트 한 문장을 추천.
    LLM 비활성화 시 None 반환.
    """
    from app.models.style_presets import get_style_prompt

    style_desc = get_style_prompt(style_key)
    if user_hint and user_hint.strip():
        user_text = f" 사용자 요청: {user_hint.strip()}"
    else:
        user_text = ""
    system = (
        "You are a short prompt writer for an image-to-image AI. "
        "Reply with exactly one short English phrase (comma-separated keywords OK) "
        "suitable as an image generation prompt. No explanation, no quotes."
    )
    user_msg = (
        f"Style: {style_desc}.{user_text} "
        "Generate one short image prompt (English, one phrase only)."
    )
    out = await complete(
        [{"role": "system", "content": system}, {"role": "user", "content": user_msg}],
        max_tokens=120,
        temperature=0.5,
    )
    return out
