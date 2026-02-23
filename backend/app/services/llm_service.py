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


class LLMQueueTimeoutError(Exception):
    """LLM 대기열 대기 시간 초과(동시 사용자 많음)."""
    pass

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
    model = None
    if device == "cuda" and getattr(settings, "llm_use_flash_attention", True):
        for candidate in ("flash_attention_2", "sdpa"):
            try:
                model = AutoModelForCausalLM.from_pretrained(
                    model_id,
                    torch_dtype=dtype,
                    device_map="auto",
                    trust_remote_code=True,
                    token=token,
                    attn_implementation=candidate,
                )
                logger.info("LLM attention: %s (H100 등에서 추론 가속)", candidate)
                break
            except Exception as e:
                logger.debug("LLM attn %s not available: %s", candidate, e)
                model = None
    if model is None:
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


# 디코딩 안정화 (BOS 이후 토큰 분포 붕괴 완화)
_GEN_TEMPERATURE = 0.6
_GEN_TOP_P = 0.85
_GEN_REPETITION_PENALTY = 1.2
_GEN_NO_REPEAT_NGRAM_SIZE = 5
_GEN_MIN_NEW_TOKENS = 50
_GEN_MAX_NEW_TOKENS = 512

# 응답 시작 검증: 한글/숫자로 시작, 경고문으로 시작 금지
_INVALID_FIRST_CHARS = set(".,;:!?\"'•·-* \t\n")
_PUNCT_AND_SYMBOLS = set(".,;:!?\"'•·-~()[]{}*#@ \t\n")


def is_valid_start(text: str) -> bool:
    """시작이 한글/숫자이고 '정확한 판단은'으로 시작하지 않으면 True."""
    if not text or not text.strip():
        return False
    first_char = text.strip()[0]
    if first_char in _INVALID_FIRST_CHARS:
        return False
    if text.strip().startswith("정확한 판단은"):
        return False
    return True


def _strip_trailing_punct_per_line(text: str) -> str:
    """줄 끝의 구두점·기호·공백 연속 제거 (한글/숫자 뒤 노이즈)."""
    lines = text.split("\n")
    out = []
    for line in lines:
        s = line.rstrip()
        # 끝에서 한글/숫자 나올 때까지 제거
        while len(s) > 0 and s[-1] in ".,;:!?\"'•·-* \t\u3000":
            s = s[:-1].rstrip()
        out.append(s)
    return "\n".join(out)


def _response_start_cleanup(text: str) -> str:
    """응답 시작 강제 정제: 첫 글자 한글/숫자까지 제거, 노이즈 줄 삭제, 줄 끝 구두점 제거."""
    if not text or not text.strip():
        return text
    s = text.strip()
    # 첫 글자가 한글/숫자가 아니면 제거 (나올 때까지)
    for i, c in enumerate(s):
        if "\uac00" <= c <= "\ud7a3" or c.isdigit():
            s = s[i:].strip()
            break
    else:
        return ""
    # 줄 단위: 끝 구두점 노이즈 제거
    s = _strip_trailing_punct_per_line(s)
    lines = s.split("\n")
    # 첫 줄: 한글 문장 뒤 구두점 난사(공백·점 3연속 이상) 나오면 그 앞까지만 유지
    if lines:
        first = lines[0]
        korean_seen = 0
        cut = len(first)
        for i, c in enumerate(first):
            if "\uac00" <= c <= "\ud7a3" or c.isdigit():
                korean_seen += 1
            elif korean_seen >= 5:
                if re.match(r"[\s.,;:!?\"'•·\-*]{3,}", first[i:]):
                    cut = i
                    break
        if cut < len(first):
            lines[0] = first[:cut].rstrip().rstrip(".,;:!?\"'•·-*")
    # 첫 줄 15자 미만·한글 거의 없음 줄 건너뛰기 (유효한 첫 줄까지)
    start = 0
    for i, line in enumerate(lines):
        stripped = line.strip()
        korean_count = len(re.findall(r"[\uac00-\ud7a3]", stripped))
        punct_count = sum(1 for c in stripped if c in ".,;:!?\"'•·-* \t")
        if len(stripped) >= 15 and korean_count >= 8 and (korean_count >= punct_count or punct_count < 10):
            start = i
            break
    s = "\n".join(lines[start:]).strip()
    return _strip_trailing_punct_per_line(s)


def _get_bad_words_ids(tokenizer: Any) -> list[list[int]]:
    """연속 구두점/따옴표/기호 시작 억제용. 같은 문자가 2회 연속 나오는 시퀀스 금지."""
    bad: list[list[int]] = []
    for char in [".", ",", ":", ";", '"', "•"]:
        ids = tokenizer.encode(char, add_special_tokens=False)
        if ids:
            for tid in ids:
                bad.append([tid, tid])
    return bad


def _run_local_inference_sync(
    messages: list[dict[str, str]],
    max_tokens: int = 256,
    temperature: float = 0.7,
) -> str | None:
    """generate_with_retry: clean_output → response_start_cleanup → is_valid_start 검증, 최대 3회 재생성."""
    try:
        model, tokenizer = _load_local_model_sync()
    except Exception as e:
        logger.exception("Local LLM load failed: %s", e)
        return None

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

    max_retry = 3
    bad_words_ids = _get_bad_words_ids(tokenizer)
    max_new = min(max(max_tokens, _GEN_MIN_NEW_TOKENS), _GEN_MAX_NEW_TOKENS)
    last_out: str | None = None
    eos_id = tokenizer.eos_token_id
    pad_id = tokenizer.pad_token_id or eos_id

    for attempt in range(max_retry):
        try:
            inputs = tokenizer(text, return_tensors="pt").to(model.device)
            gen = model.generate(
                **inputs,
                min_new_tokens=_GEN_MIN_NEW_TOKENS,
                max_new_tokens=max_new,
                do_sample=True,
                temperature=_GEN_TEMPERATURE,
                top_p=_GEN_TOP_P,
                repetition_penalty=_GEN_REPETITION_PENALTY,
                no_repeat_ngram_size=_GEN_NO_REPEAT_NGRAM_SIZE,
                pad_token_id=pad_id,
                eos_token_id=eos_id,
                bad_words_ids=bad_words_ids if bad_words_ids else None,
            )
            raw = tokenizer.decode(gen[0][inputs.input_ids.shape[1] :], skip_special_tokens=True)
            cleaned = clean_output(raw.strip() or "")
            cleaned = _response_start_cleanup(cleaned)
            last_out = cleaned
            if not cleaned:
                continue
            if is_valid_start(cleaned):
                return cleaned
            logger.debug("Retry generate attempt %s: invalid start", attempt + 1)
        except Exception as e:
            logger.warning("Local LLM inference failed (attempt %s): %s", attempt + 1, e)
    return clean_output(last_out) if last_out else None


# 고정 문구 강박 차단: 1회 초과 시 제거, 마지막에 1회만 허용
_FIXED_PHRASE_BLOCK = (
    "정확한 판단은",
    "전문가에게 확인하세요",
    "수의사와 상담",
    "의료·수의 전문가",
)


def _strip_fixed_phrase_repeats(text: str) -> str:
    """고정 경고 문구가 1회 초과면 마지막 1회만 유지."""
    if not text or not text.strip():
        return text
    s = text
    for phrase in _FIXED_PHRASE_BLOCK:
        count = s.count(phrase)
        if count <= 1:
            continue
        idx = 0
        last_start = -1
        while True:
            pos = s.find(phrase, idx)
            if pos == -1:
                break
            last_start = pos
            idx = pos + 1
        if last_start == -1:
            continue
        before = s[:last_start]
        after = s[last_start:]
        before_removed = before.replace(phrase, " ").strip()
        s = (before_removed + "\n\n" + after.strip()).strip() if before_removed else after.strip()
    return s


def _punctuation_density(s: str) -> float:
    """구두점·특수기호 비율 (0~1)."""
    if not s or not s.strip():
        return 0.0
    punct = sum(1 for c in s if c in ".,;:!?\"'•·-~()[]{}*#@")
    return punct / max(len(s), 1)


def clean_output(text: str) -> str:
    """
    추론 출력 정화: 연속 구두점/줄 시작 특수문자/경고문 중복 제거,
    첫 줄 정제, 동일 문장 2회 이상 → 1회, 구두점 밀도 정규화.
    """
    if not text or not text.strip():
        return text
    s = text.strip()
    # 따옴표 조각 제거: "ㄱㅏㅇ"? " " "조언" "니" "※" 등
    s = re.sub(r'"\s*[^"]{1,6}\s*"\s*\??', " ", s)
    s = re.sub(r'"\s*"\s*', " ", s)
    s = re.sub(r"'\s*'\s*'", " ", s)
    # 연속 구두점 제거 (2회 이상 → 제거)
    s = re.sub(r"[.,:;\"'\-]{2,}", " ", s)
    # 전체/줄 시작 특수문자 제거
    s = re.sub(r"^[\s.,:;\"'\-•*]+", "", s)
    s = re.sub(r"\n[\s.,:;\"'\-•*]+", "\n", s)
    # 고정 경고문 2회 이상 → 1회
    s = re.sub(
        r"(정확한 판단은 의료·수의 전문가에게 확인하세요\.?){2,}",
        r"정확한 판단은 의료·수의 전문가에게 확인하세요.",
        s,
    )
    # 빈 줄 3개 이상 → 2줄
    s = re.sub(r"\n{3,}", "\n\n", s)
    s = s.strip()
    # 응답 첫 줄이 특수문자로 시작하면 제거 (한글/숫자 나올 때까지)
    lines = s.split("\n")
    start = 0
    for i, line in enumerate(lines):
        stripped = line.strip()
        if not stripped:
            start = i + 1
            continue
        first = stripped[0]
        if first in _PUNCT_AND_SYMBOLS or (len(stripped) < 2 and stripped in ".,;:!?\"'•"):
            start = i + 1
            continue
        if re.search(r"[\uac00-\ud7a3\d]", stripped):
            start = i
            break
        start = i + 1
    s = "\n".join(lines[start:]).strip()
    # 연속 특수문자 2개 이상 제거 (한 칸으로)
    s = re.sub(r"([.,;:!?\"'•·\-~()\[\]{}*#@\s])\s*\1+", r"\1", s)
    # 불필요한 따옴표 전체 제거 (짧은 인용만 있는 경우)
    s = re.sub(r'"\s*"+', " ", s)
    s = re.sub(r"^\s*[\"']+\s*", "", s)
    # 줄 시작이 기호면 해당 줄 삭제; 구두점만/한글 3글자 미만 노이즈 줄 삭제; 기호 위주 줄 삭제
    out_lines = []
    for line in s.split("\n"):
        stripped = line.strip()
        if stripped and stripped[0] in ".,;:!?\"'•·-*#@•":
            continue
        if stripped and _is_noise_line(stripped):
            continue
        korean = len(re.findall(r"[\uac00-\ud7a3]", stripped))
        punct = sum(1 for c in stripped if c in ".,;:!?\"'•·-* \t")
        if len(stripped) >= 4 and korean < 3 and punct >= 2:
            continue
        out_lines.append(line)
    s = "\n".join(out_lines)
    s = _strip_trailing_punct_per_line(s)
    # 연속 구두점 2회 이상 → 1회
    s = re.sub(r"([.,;:!?\-~])\s*\1+", r"\1", s)
    # 공백 3회 이상 → 1회
    s = re.sub(r" {3,}", " ", s)
    # 빈 줄 3줄 이상 → 2줄
    s = re.sub(r"\n{3,}", "\n\n", s)
    # 동일 문장 2회 이상 제거 (줄 단위)
    lines = s.split("\n")
    seen = set()
    out = []
    for line in lines:
        key = line.strip()[:80] if line.strip() else ""
        if key and key in seen:
            continue
        if key:
            seen.add(key)
        out.append(line)
    s = "\n".join(out)
    # 고정 문구 1회 초과 제거 (마지막 1회만 유지)
    s = _strip_fixed_phrase_repeats(s)
    # 구두점 밀도 15% 초과 시 정규화: 연속 구두점·줄 앞 구두점 제거
    if _punctuation_density(s) > 0.15:
        s = re.sub(r"([.,;:!?])\s*\1+", r"\1", s)
        s = re.sub(r"\n\s*[.,;:!?]+\s*", "\n", s)
    # JSON 조각 제거
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

    try:
        async with httpx.AsyncClient(timeout=settings.llm_timeout_seconds) as client:
            r = await client.post(url, json=payload, headers=headers)
            r.raise_for_status()
            data = r.json()
    except httpx.ConnectError as e:
        logger.warning(
            "LLM API 연결 실패 (vLLM 서버 확인): URL=%s, 오류=%s. "
            "같은 호스트면 http://127.0.0.1:7001/v1, 다른 Pod/호스트면 해당 주소로 LLM_API_BASE 설정.",
            url,
            e,
        )
        return None
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
    로컬 모델은 한 번에 하나만 실행(_local_lock). 대기 초과 시 LLMQueueTimeoutError → 503.
    """
    if not is_llm_available():
        return None
    settings = get_settings()
    sem = _get_llm_semaphore()
    try:
        await asyncio.wait_for(sem.acquire(), timeout=settings.llm_queue_wait_seconds)
    except asyncio.TimeoutError:
        logger.warning(
            "LLM queue full: wait timeout (%ss). 다른 사용자 사용 중.",
            settings.llm_queue_wait_seconds,
        )
        raise LLMQueueTimeoutError(
            f"다른 사용자가 사용 중입니다. {settings.llm_queue_wait_seconds}초 후 다시 시도해 주세요."
        )

    try:
        if settings.llm_use_local:
            # 로컬 모델은 동시에 한 추론만 허용 (GPU/스레드 충돌 방지)
            try:
                await asyncio.wait_for(
                    _local_lock.acquire(), timeout=settings.llm_queue_wait_seconds
                )
            except asyncio.TimeoutError:
                sem.release()
                logger.warning(
                    "로컬 LLM 대기 시간 초과 (%ss). 다른 사용자 사용 중.",
                    settings.llm_queue_wait_seconds,
                )
                raise LLMQueueTimeoutError(
                    f"다른 사용자가 사용 중입니다. {settings.llm_queue_wait_seconds}초 후 다시 시도해 주세요."
                )
            try:
                loop = asyncio.get_event_loop()
                return await loop.run_in_executor(
                    None,
                    lambda: _run_local_inference_sync(messages, max_tokens, temperature),
                )
            finally:
                _local_lock.release()
        try:
            return await _complete_via_api(messages, max_tokens, temperature)
        except Exception as e:
            logger.warning("LLM API request failed: %s", e)
            return None
    finally:
        sem.release()


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
    "넣으라고",
    "이렇게 나오는데",
    "표현금지",
    "표현 금지",
    "정확한 판단 =",
    "정확한 판단\"",
    "의료·수와 전문가",
    "의료·수를 통해",
    "확인하시길 바랍니다",
    "전공자가 검토",
    "전공자에게 확인",
    "진단·확정",
    "답변 맨 끝",
    "답변 맨끝",
    "답변 맹엔",
    "한 번만 썬다",
    "한 번만 썰라",
    "한 번만 쓴다",
    "정확히 한 번만",
    "정확적으로 한 번만",
    "수의사와 상의",
    "상의하시길 바랍니다",
    "과정을 거쳐야",
    "정확히 판단하기 위해서는",
    "정확한 판단을 위해",
    "지시문",
    "메타 설명",
)


def _is_disclaimer_variant_line(s: str) -> bool:
    """면책 문구 변형·오타 한 줄이면 True. (제거 후 마지막에 정식 문장 1회만 붙임)"""
    t = s.strip()
    if len(t) > 80:
        return False
    norm = re.sub(r"[^\uac00-\ud7a3·]", "", t)
    # 정확한판단 + 의료/수의 + 확인/바랍 등 핵심만 있는 짧은 줄
    if "정확한" in norm and ("의료" in norm or "수의" in norm) and ("확인" in norm or "바랍" in norm or "검토" in norm or "상의" in norm):
        return True
    if re.match(r"^[\s\.\:\'\"]*정확한\s*판단", t) and len(norm) < 35:
        return True
    # "전문가에게 확인하세요" 단독 또는 앞뒤 기호만
    if re.search(r"전문가에게\s*확인", t) and len(norm) < 30:
        return True
    if re.search(r"수의사와\s*상의", t) or re.search(r"전공자에게\s*확인", t):
        return True
    return False


def _is_noise_line(s: str) -> bool:
    """기호·구두점 위주 줄(한글 거의 없음)이면 True. 제거 대상."""
    t = s.strip()
    if not t or len(t) > 200:
        return False
    korean = len(re.findall(r"[가-힣]", t))
    punct_etc = sum(1 for c in t if c in ".,:;!?\"'•·\-~()[]{}*#@ \t\n\u3000")
    # 한글이 3글자 미만이고 기호/공백이 절반 이상
    if korean < 3 and punct_etc >= len(t) // 2:
        return True
    # 줄 전체가 따옴표·콜론·공백 조합 (예: "   '  :    : " ")
    if korean == 0 and len(t) >= 3:
        return True
    return False


HEALTH_ASSISTANT_SYSTEM_PROMPT = """당신은 반려동물 건강 질문에 답하는 도우미입니다. 참고 정보만 안내하고, 진단·확정 표현은 하지 않습니다.

응답 규칙:
- 한글만 사용. 영문·기호로 시작하지 말고, 바로 본문 문장으로 시작.
- 소제목은 **굵게**, 목록은 "• 항목" 한 줄씩.
- 답변 마지막에 반드시 이 한 문장만 붙임: 정확한 판단은 의료·수의 전문가에게 확인하세요.
- 위 문장 외에 지시문, 인용부호 난입, 반복 금지. 반려동물 건강 질문이 아니면 답하지 않음.
- 약물·처방·복용·투여·경구·주사·mg 등 약과 관련한 언급은 절대 하지 않습니다. "병원에서 확인 후 처방받으세요" 같은 문장도 쓰지 마세요. 질문을 이어갈 수 있도록 다음에 물어볼 만한 것을 안내해 주세요."""

# 구조화 건강 상담: JSON만 출력, 4순위 필수, 감별 2~3문장·병태생리·관찰 포인트 필수
HEALTH_ASSISTANT_STRUCTURED_PROMPT = """당신은 반려동물 질병 상담 AI입니다. 임상 추론 수준을 중급 이상으로 유지하세요.

[절대 규칙]
- 절대 한 줄로 끝내지 말 것. 모든 답변은 반드시 아래 JSON 구조로만 출력한다. JSON 외 본문·설명·소제목을 쓰지 마세요.
- 감별 진단은 4순위까지 반드시 모두 출력. 1~3개만 쓰면 오류로 간주.
- 각 감별 진단(differential 항목)은 최소 2~3문장으로 설명. reason과 home_check에 한두 단어만 쓰면 오류.
- 병태생리 연결 필수: reason에는 "왜 이 증상이 이 질환과 연결되는지" 병태생리적 근거를 2~3문장으로 쓸 것.
- 보호자가 이해할 수 있는 관찰 포인트 필수: home_check에는 집에서 바로 확인할 수 있는 행동·징후를 구체적으로 쓸 것.
- 설명 누락 시 오류로 간주됨.

[금지]
- 단순 질환 나열·한 줄 요약 금지. 5순위 이상 출력 금지.
- 추상적 표현 금지 (예: "혈압이 낮으면" X). 행동 기준으로 명확히.
- 약물·처방·복용·투여·mg 등 약 관련 언급 금지. 이어갈 질문(recommended_categories)으로 보호자를 도와주세요.

[출력 형식 - 이 JSON만 출력]
응답 전체는 반드시 아래 형식의 JSON 블록 하나만 출력하세요. ```json 앞뒤에 다른 글 없이, 블록만 출력.

```json
{
  "differential": [
    { "rank": 1, "name": "질환명", "reason": "해당 증상과 이 질환이 연결되는 병태생리적 이유를 2~3문장으로 설명. 한 줄 금지.", "emergency": true 또는 false, "home_check": "보호자가 집에서 확인할 수 있는 관찰 포인트를 2~3문장으로." },
    { "rank": 2, "name": "질환명", "reason": "2~3문장 병태생리 근거.", "emergency": false, "home_check": "2~3문장 관찰 포인트." },
    { "rank": 3, "name": "질환명", "reason": "2~3문장 병태생리 근거.", "emergency": false, "home_check": "2~3문장 관찰 포인트." },
    { "rank": 4, "name": "질환명", "reason": "2~3문장 병태생리 근거.", "emergency": false, "home_check": "2~3문장 관찰 포인트." }
  ],
  "emergency_criteria": ["행동 기준 1", "행동 기준 2", "..."],
  "key_questions": ["감별에 필요한 질문 1", "질문 2", "최대 5개"],
  "recommended_categories": [
    { "label": "카테고리 라벨", "query": "보호자가 바로 물어볼 수 있는 질문 문장" },
    { "label": "...", "query": "..." }
  ]
}
```

- differential은 반드시 4개. rank 1,2,3,4 모두 포함. reason·home_check는 각각 최소 2~3문장 분량.
- emergency는 반드시 true 또는 false. recommended_categories는 최소 2개 이상."""

# 약물·처방 언급 제거용 패턴 (한 줄이 이 키워드 위주면 해당 줄 제거)
_MEDICATION_LINE_PATTERN = re.compile(
    r"(?:약\s*(?:물|이름|을|을\s*먹|복용|투여)|처방|복용|투여|경구|주사|정제|캡슐|mg\s*이상|mL\s*이상|용량|내복|외용)"
)


def _strip_medication_mentions(text: str) -> str:
    """약물·처방 관련 문장이 포함된 줄을 제거. JSON 블록 앞 본문만 처리."""
    if not text or not text.strip():
        return text
    # ```json 이전 부분만 처리 (JSON 블록은 그대로 유지)
    parts = re.split(r"(\s*```(?:json)?\s*[\s\S]*?\s*```\s*)", text, maxsplit=1)
    narrative = parts[0]
    rest = parts[1] if len(parts) > 1 else ""
    lines = narrative.split("\n")
    kept = []
    for line in lines:
        strip_line = line.strip()
        if not strip_line:
            if kept:
                kept.append(line)
            continue
        if _MEDICATION_LINE_PATTERN.search(strip_line):
            continue
        kept.append(line)
    result = "\n".join(kept).strip()
    if rest:
        result = result + "\n" + rest
    return re.sub(r"\n{3,}", "\n\n", result).strip()


def _extract_health_structured(raw: str) -> Any | None:
    """응답 텍스트에서 ```json ... ``` 블록을 찾아 HealthChatStructured로 파싱. 실패 시 None."""
    import json
    from app.schemas.health_chat_schema import HealthChatStructured

    if not raw or not raw.strip():
        return None
    # ```json ... ``` 또는 ``` ... ```
    match = re.search(r"```(?:json)?\s*([\s\S]*?)\s*```", raw)
    if not match:
        return None
    try:
        data = json.loads(match.group(1).strip())
        return HealthChatStructured(**data)
    except Exception:
        return None


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
    """알파벳 제거 후 남은 앞쪽 구두점·쓰레기·면책 변형·지시 유출 줄 제거. 본문(한글/•/**)만 남김."""
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
        # 기호·구두점 위주 노이즈 줄 제거
        if _is_noise_line(s):
            continue
        # 앞쪽에 나온 면책 변형·지시 유출 짧은 줄은 제거 (본문이 나오기 전까지)
        if len(s) < 70 and (_is_disclaimer_variant_line(s) or _is_instruction_leakage(s)):
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
    """반복 문구·disclaimer 변형·지시 유출 제거. disclaimer는 맨 끝에 정식 문장 한 번만."""
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
        # 면책 문구 변형·오타 줄 제거 (예: 정확한 판단는, 확인하시길 바랍니다 등)
        if _is_disclaimer_variant_line(s):
            continue
        if _is_instruction_leakage(s):
            continue
        # 기호·구두점 위주 노이즈 줄 제거
        if _is_noise_line(s):
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
    # 앞쪽 노이즈: disclaimer/지시 유출만 있는 줄들이 먼저 나온 경우 제거했으므로, 본문만 남음.
    # 맨 끝에 정식 면책 문장 1회만
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
) -> tuple[str | None, Any | None]:
    """건강 질문 도우미(구조화). (본문 텍스트, HealthChatStructured | None) 반환."""
    if not messages or (messages and (messages[0].get("role") or "").lower() != "system"):
        msgs = [{"role": "system", "content": HEALTH_ASSISTANT_STRUCTURED_PROMPT}] + list(messages)
    else:
        msgs = [{"role": "system", "content": HEALTH_ASSISTANT_STRUCTURED_PROMPT}] + [
            m for m in messages if (m.get("role") or "").lower() != "system"
        ]
    result = await complete(msgs, max_tokens=max_tokens, temperature=temperature)
    if not result:
        return (result, None)
    if _last_user_content_has_hangul(msgs):
        result = _strip_alphabet(result)
        if result:
            result = _dedupe_and_fix_disclaimer(result)
    result = _strip_medication_mentions(result)
    structured = _extract_health_structured(result)
    if structured is not None:
        narrative = re.sub(r"\s*```(?:json)?\s*[\s\S]*?\s*```\s*$", "", result).strip()
        result = narrative if narrative else "감별 진단과 관찰 포인트를 확인하세요. 정확한 판단은 의료·수의 전문가에게 확인하세요."
    return (result, structured)


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
