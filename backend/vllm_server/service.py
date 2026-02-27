"""
vLLM inference service: concurrency control, queueing, timeout, streaming.
Uses backend URL (vLLM OpenAI server) or embedded engine. No blocking in event loop.
"""

from __future__ import annotations

import asyncio
import logging
import time
from typing import AsyncIterator

import httpx

from vllm_server.config import get_vllm_settings
from vllm_server.schemas import (
    ChatCompletionRequest,
    ChatCompletionResponse,
    ChatCompletionChoice,
    ChatMessage,
)

logger = logging.getLogger(__name__)


def _mock_completion_response(req: ChatCompletionRequest) -> ChatCompletionResponse:
    """OpenAI 형식 mock 응답 (vLLM 없이 테스트용). 메인 앱 clean_output이 기호로 시작하는 줄을 제거하므로 한글로 시작."""
    last_content = req.messages[-1].content if req.messages else ""
    reply = f"테스트 모드 응답입니다. 요청하신 내용: {last_content[:80]}{'...' if len(last_content) > 80 else ''}"
    return ChatCompletionResponse(
        id="mock-cmpl-1",
        object="chat.completion",
        created=int(time.time()),
        model=req.model,
        choices=[
            ChatCompletionChoice(
                index=0,
                message=ChatMessage(role="assistant", content=reply),
                finish_reason="stop",
            )
        ],
        usage={"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0},
    )


async def _mock_completion_stream(req: ChatCompletionRequest) -> AsyncIterator[str]:
    """SSE 스트리밍 mock (vLLM 없이 테스트용)."""
    import json
    last_content = req.messages[-1].content if req.messages else ""
    reply = f"테스트 모드 응답입니다. {last_content[:50]}{'...' if len(last_content) > 50 else ''} 에 대한 답변입니다."
    chunk = {
        "id": "mock-stream-1",
        "object": "chat.completion.chunk",
        "created": int(time.time()),
        "model": req.model,
        "choices": [{"index": 0, "delta": {"role": "assistant", "content": reply}, "finish_reason": None}],
    }
    yield f"data: {json.dumps(chunk, ensure_ascii=False)}\n\n"
    yield "data: [DONE]\n\n"


class VLLMQueueTimeoutError(Exception):
    """Raised when waiting for a request slot exceeds queue_wait_timeout_seconds."""

    def __init__(self, message: str) -> None:
        self.message = message
        super().__init__(message)


class VLLMService:
    """
    Production service: semaphore-based concurrency, async HTTP to vLLM backend.
    No global blocking locks; no synchronous generate in event loop.
    """

    def __init__(self) -> None:
        self._semaphore: asyncio.Semaphore | None = None
        self._client: httpx.AsyncClient | None = None
        self._closed = False

    def _get_semaphore(self) -> asyncio.Semaphore:
        if self._semaphore is None:
            settings = get_vllm_settings()
            self._semaphore = asyncio.Semaphore(settings.max_concurrent_requests)
        return self._semaphore

    def _get_client(self) -> httpx.AsyncClient:
        if self._client is None or self._closed:
            settings = get_vllm_settings()
            self._client = httpx.AsyncClient(
                base_url=settings.backend_url.rstrip("/"),
                timeout=httpx.Timeout(settings.request_timeout_seconds),
            )
            self._closed = False
        return self._client

    async def close(self) -> None:
        """Release resources. Safe to call multiple times."""
        self._closed = True
        if self._client is not None:
            await self._client.aclose()
            self._client = None

    def _build_body(self, req: ChatCompletionRequest) -> dict:
        return {
            "model": req.model,
            "messages": [{"role": m.role, "content": m.content} for m in req.messages],
            "stream": req.stream,
            "max_tokens": req.max_tokens,
            "temperature": req.temperature,
            "top_p": req.top_p,
            "stop": req.stop,
        }

    async def chat_completion(self, req: ChatCompletionRequest) -> ChatCompletionResponse:
        """
        Non-streaming completion. Acquires semaphore with queue timeout, then POST to vLLM (or mock).
        """
        settings = get_vllm_settings()
        sem = self._get_semaphore()
        try:
            await asyncio.wait_for(
                sem.acquire(),
                timeout=settings.queue_wait_timeout_seconds,
            )
        except asyncio.TimeoutError:
            raise VLLMQueueTimeoutError(
                f"서버가 바쁩니다. {int(settings.queue_wait_timeout_seconds)}초 후 다시 시도해 주세요."
            )
        try:
            if settings.use_mock:
                logger.debug("Mock completion (no vLLM backend)")
                return _mock_completion_response(req)
            if not (settings.backend_url or "").strip():
                raise VLLMQueueTimeoutError(
                    "VLLM 백엔드가 없습니다. VLLM_USE_MOCK=1 (테스트) 또는 VLLM_BACKEND_URL=http://127.0.0.1:8000 (vllm serve 포트) 설정 후 재실행하세요."
                )
            client = self._get_client()
            body = self._build_body(req)
            start = time.perf_counter()
            response = await client.post(
                "/v1/chat/completions",
                json=body,
            )
            response.raise_for_status()
            data = response.json()
            elapsed = time.perf_counter() - start
            logger.info(
                "vLLM chat completion OK, %.2fs, choices=%s",
                elapsed,
                len(data.get("choices", [])),
            )
            return ChatCompletionResponse(**data)
        finally:
            sem.release()

    async def chat_completion_stream(
        self,
        req: ChatCompletionRequest,
    ) -> AsyncIterator[str]:
        """
        Streaming completion: yields SSE lines (data: {...}). Acquires semaphore with timeout.
        """
        settings = get_vllm_settings()
        sem = self._get_semaphore()
        try:
            await asyncio.wait_for(
                sem.acquire(),
                timeout=settings.queue_wait_timeout_seconds,
            )
        except asyncio.TimeoutError:
            raise VLLMQueueTimeoutError(
                f"서버가 바쁩니다. {int(settings.queue_wait_timeout_seconds)}초 후 다시 시도해 주세요."
            )
        try:
            if settings.use_mock:
                logger.debug("Mock stream (no vLLM backend)")
                async for chunk in _mock_completion_stream(req):
                    yield chunk
                return
            if not (settings.backend_url or "").strip():
                raise VLLMQueueTimeoutError(
                    "VLLM 백엔드가 없습니다. VLLM_USE_MOCK=1 (테스트) 또는 VLLM_BACKEND_URL=http://127.0.0.1:8000 (vllm serve 포트) 설정 후 재실행하세요."
                )
            client = self._get_client()
            body = self._build_body(req)
            async with client.stream(
                "POST",
                "/v1/chat/completions",
                json=body,
                timeout=httpx.Timeout(settings.request_timeout_seconds),
            ) as response:
                response.raise_for_status()
                async for line in response.aiter_lines():
                    if line.startswith("data: "):
                        yield line + "\n"
                    if line.strip() == "data: [DONE]":
                        yield "data: [DONE]\n\n"
                        break
        finally:
            sem.release()


# Optional: embedded engine (in-process AsyncLLM). Used when backend_url is not set.
def _create_embedded_engine():  # type: ignore[no-untyped-def]
    """Create AsyncLLM engine. Lazy import to avoid requiring vLLM when using gateway only."""
    try:
        from vllm.engine.arg_utils import AsyncEngineArgs
        from vllm.v1.engine.async_llm import AsyncLLM
    except ImportError:
        try:
            from vllm import AsyncLLMEngine
            from vllm.engine.arg_utils import AsyncEngineArgs
            AsyncLLM = AsyncLLMEngine  # type: ignore[misc, assignment]
        except ImportError:
            raise ImportError("vLLM not installed. Install with: pip install vllm") from None

    settings = get_vllm_settings()
    kwargs: dict = {
        "model": settings.model_id,
        "gpu_memory_utilization": settings.gpu_memory_utilization,
        "max_num_seqs": settings.max_num_seqs,
        "tensor_parallel_size": settings.tensor_parallel_size,
        "trust_remote_code": settings.trust_remote_code,
        "enforce_eager": settings.enforce_eager,
    }
    if settings.max_model_len is not None:
        kwargs["max_model_len"] = settings.max_model_len
    if settings.dtype != "auto":
        kwargs["dtype"] = settings.dtype
    if settings.hf_token:
        kwargs["hf_token"] = settings.hf_token
    engine_args = AsyncEngineArgs(**kwargs)
    return AsyncLLM.from_engine_args(engine_args)


# Singleton service instance (no global blocking locks; semaphore is async)
_service: VLLMService | None = None


def get_vllm_service() -> VLLMService:
    global _service
    if _service is None:
        _service = VLLMService()
    return _service
