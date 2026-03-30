"""
LTX-Video / LTX-2 계열 공통 제약.

- HF LTX-Video 카드: ``num_frames`` 는 보통 ``8n+1`` (9, 17, …, 49, …).
- 해상도: LTX-Video 예시는 8의 배수, 이 프로젝트의 LTX-2 경로는 32배수를 사용 (``video_service``).

실제 I2V 추론은 ``app.services.video_service`` 에서 수행한다.
"""

from __future__ import annotations


def clamp_wh_multiple_of_8(width: int, height: int, w_min: int = 256, w_max: int = 1280) -> tuple[int, int]:
    """width/height를 8의 배수로 맞춘다 (Lightricks LTX-Video 문서 권장 패턴)."""
    w = max(w_min, min(w_max, (max(width, 8) // 8) * 8))
    h = max(w_min, min(w_max, (max(height, 8) // 8) * 8))
    return w, h


def clamp_wh_multiple_of_32(width: int, height: int, w_min: int = 256, w_max: int = 1280) -> tuple[int, int]:
    """LTX-2.3 / 현재 ``video_service`` 와 동일하게 32배수."""
    w = max(w_min, min(w_max, (max(width, 32) // 32) * 32))
    h = max(w_min, min(w_max, (max(height, 32) // 32) * 32))
    return w, h


def clamp_frames_8n_plus_1(num_frames: int, min_frames: int = 9, max_frames: int = 241) -> int:
    """가장 가까운 ``8n+1`` 프레임 수로 보정."""
    if num_frames <= 0:
        return max(min_frames, 9)
    n = max(min_frames, min(max_frames, num_frames))
    r = (n - 1) % 8
    if r == 0:
        return n
    low = ((n - 1) // 8) * 8 + 1
    high = min(max_frames, low + 8)
    return low if (n - low) <= (high - n) else high
