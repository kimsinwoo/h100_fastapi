"""
FastAPI routes: /v1/chat/completions (stream + non-stream), validation, 503 on queue timeout.
"""

from __future__ import annotations

import logging
from typing import AsyncIterator

from fastapi import APIRouter, HTTPException
from fastapi.responses import StreamingResponse

from vllm_server.schemas import ChatCompletionRequest
from vllm_server.service import VLLMQueueTimeoutError, get_vllm_service

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/v1", tags=["chat"])


async def _sse_stream(generator: AsyncIterator[str]) -> AsyncIterator[str]:
    """Relay SSE chunks from backend to client."""
    async for chunk in generator:
        yield chunk


@router.post("/chat/completions", response_model=None)
async def chat_completions(body: ChatCompletionRequest):
    """
    OpenAI-compatible chat completions. Use stream=true for SSE streaming.
    Concurrency and queue timeout enforced by service layer.
    """
    service = get_vllm_service()
    try:
        if not body.stream:
            result = await service.chat_completion(body)
            return result
        # Streaming: return SSE
        return StreamingResponse(
            _sse_stream(service.chat_completion_stream(body)),
            media_type="text/event-stream",
            headers={
                "Cache-Control": "no-cache",
                "Connection": "keep-alive",
                "X-Accel-Buffering": "no",
            },
        )
    except VLLMQueueTimeoutError as e:
        logger.warning("Chat completions queue timeout: %s", e.message)
        raise HTTPException(status_code=503, detail=e.message) from e
    except Exception as e:
        logger.exception("Chat completions error: %s", e)
        raise HTTPException(status_code=503, detail="LLM request failed") from e


@router.get("/health")
async def health() -> dict[str, str]:
    return {"status": "ok"}
