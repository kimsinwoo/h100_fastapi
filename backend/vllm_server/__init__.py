"""
Production vLLM gateway/embedded server.
Handles 100+ concurrent users via queueing, async, and no event-loop blocking.
"""

from vllm_server.main import create_app

__all__ = ["create_app"]
