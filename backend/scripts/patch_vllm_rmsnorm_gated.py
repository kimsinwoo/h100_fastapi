#!/usr/bin/env python3
"""
vLLM Qwen3.5-35B-A3B 실행 시 발생하는
  AttributeError: 'RMSNormGated' object has no attribute 'activation'
를 우회하기 위한 패치.

vLLM model_executor/layers/layernorm.py 의 RMSNormGated.forward_cuda 에서
self.activation 을 참조하는데, CustomOp 쪽 RMSNormGated 는 __init__ 에 activation 이 없음.
rmsnorm_fn 의 activation 기본값은 'swish'(=silu) 이므로, 없으면 'silu' 로 넘기면 됨.

사용법 (backend venv 활성화 후):
  python scripts/patch_vllm_rmsnorm_gated.py
"""
from __future__ import annotations

import sys
from pathlib import Path


def main() -> int:
    try:
        import vllm
    except ImportError:
        print("vllm 이 설치되어 있지 않습니다. pip install vllm 후 다시 실행하세요.", file=sys.stderr)
        return 1

    # site-packages/vllm/model_executor/layers/layernorm.py
    layernorm_path = Path(vllm.__file__).resolve().parent / "model_executor" / "layers" / "layernorm.py"
    if not layernorm_path.is_file():
        print(f"파일을 찾을 수 없습니다: {layernorm_path}", file=sys.stderr)
        return 1

    text = layernorm_path.read_text(encoding="utf-8")
    old = "activation=self.activation,"
    new = "activation=getattr(self, 'activation', 'silu'),"

    if new in text:
        print("이미 패치가 적용되어 있습니다.")
        return 0
    if old not in text:
        print(f"패치할 문자열을 찾을 수 없습니다. vLLM 버전이 다를 수 있습니다.", file=sys.stderr)
        print(f"찾는 문자열: {old!r}", file=sys.stderr)
        return 1

    layernorm_path.write_text(text.replace(old, new, 1), encoding="utf-8")
    print(f"패치 적용 완료: {layernorm_path}")
    print("  RMSNormGated.forward_cuda 에서 activation=getattr(self, 'activation', 'silu') 사용")
    return 0


if __name__ == "__main__":
    sys.exit(main())
