#!/usr/bin/env python3
"""
Run the vLLM gateway (FastAPI). Start the vLLM OpenAI server separately with H100 params.
Example:
  Terminal 1: ./scripts/start_vllm_h100.sh   # or: vllm serve openai/gpt-oss-20b ...
  Terminal 2: python run_vllm_gateway.py
"""
from __future__ import annotations

import os
import sys

# Ensure backend root is on path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import uvicorn
from vllm_server.config import get_vllm_settings
from vllm_server.main import app

if __name__ == "__main__":
    settings = get_vllm_settings()
    uvicorn.run(
        app,
        host=settings.host,
        port=settings.port,
        workers=1,
        log_level=settings.log_level.lower(),
    )
