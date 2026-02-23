#!/usr/bin/env python3
"""
vLLM 게이트웨이 Mock 모드 동작 검증.
사용법: 게이트웨이를 Mock 모드로 띄운 뒤 다른 터미널에서:
  cd zimage_webapp/backend && source venv/bin/activate && python scripts/test_vllm_gateway_mock.py
또는 한 터미널에서: VLLM_USE_MOCK=1 python run_vllm_gateway.py & sleep 2 && python scripts/test_vllm_gateway_mock.py
"""
from __future__ import annotations

import os
import sys

import httpx

GATEWAY_URL = os.environ.get("VLLM_GATEWAY_URL", "http://127.0.0.1:7001")


def main() -> int:
    print(f"Testing vLLM gateway at {GATEWAY_URL} (expect mock response)...")
    try:
        # Health
        r = httpx.get(f"{GATEWAY_URL}/v1/health", timeout=5.0)
        r.raise_for_status()
        print("  GET /v1/health OK:", r.json())

        # Chat completion (non-stream)
        r = httpx.post(
            f"{GATEWAY_URL}/v1/chat/completions",
            json={
                "model": "gpt-oss-20b",
                "messages": [{"role": "user", "content": "테스트 메시지"}],
                "stream": False,
            },
            timeout=10.0,
        )
        r.raise_for_status()
        data = r.json()
        choices = data.get("choices", [])
        if not choices:
            print("  POST /v1/chat/completions: no choices in response")
            return 1
        content = choices[0].get("message", {}).get("content", "")
        if "[Mock]" not in content:
            print("  POST /v1/chat/completions: response does not look like mock:", content[:100])
            return 1
        print("  POST /v1/chat/completions OK (mock):", content[:80] + "...")
        print("All checks passed.")
        return 0
    except httpx.ConnectError as e:
        print("  Connection failed. Is the gateway running with VLLM_USE_MOCK=1?", file=sys.stderr)
        print("  Example: cd zimage_webapp/backend && source venv/bin/activate && bash scripts/run_vllm_gateway_mock.sh", file=sys.stderr)
        return 1
    except Exception as e:
        print("  Error:", e, file=sys.stderr)
        return 1


if __name__ == "__main__":
    sys.exit(main())
