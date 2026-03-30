"""
LTX 계열 I2V — 제약 정규화·설정 헬퍼.

실제 ``diffusers`` 파이프라인 로드·추론은 ``app.services.video_service.run_image_to_video`` 에 구현되어 있다.
(모델 ID: ``Settings.ltx2_model_id``, ComfyUI 경로는 ``app.services.comfyui_service``.)

향후 ``LTXImageToVideoPipeline`` 전용 싱글턴을 두려면 이 클래스에 로드를 옮기면 된다.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from app.core.config import get_settings
from app.inference.ltx_constraints import (
    clamp_frames_8n_plus_1,
    clamp_wh_multiple_of_32,
)


@dataclass(frozen=True)
class LTXGenerationParams:
    """영상 생성에 쓸 정규화된 파라미터."""

    width: int
    height: int
    num_frames: int
    num_inference_steps: int
    guidance_scale: float
    model_id: str


class LTXVideoEngine:
    """
    LTX-Video / LTX-2 공통 제약을 적용한 파라미터 빌더.

    GPU 추론 루틴은 ``video_service`` 에 위임한다.
    """

    @staticmethod
    def normalize_params(
        width: int,
        height: int,
        num_frames: int,
        *,
        use_32_multiple: bool = True,
    ) -> tuple[int, int, int]:
        """
        해상도·프레임을 LTX 계열 규칙에 맞게 맞춘다.

        ``use_32_multiple=True`` 이면 현재 백엔드 LTX-2 경로와 동일(32배수).
        Lightricks LTX-Video 문서만 따르려면 ``clamp_wh_multiple_of_8`` 를 직접 호출하라.
        """
        if use_32_multiple:
            w, h = clamp_wh_multiple_of_32(width, height)
        else:
            from app.inference.ltx_constraints import clamp_wh_multiple_of_8

            w, h = clamp_wh_multiple_of_8(width, height)
        nf = clamp_frames_8n_plus_1(num_frames)
        return w, h, nf

    @staticmethod
    def params_from_settings(
        *,
        width: int | None = None,
        height: int | None = None,
        num_frames: int | None = None,
    ) -> LTXGenerationParams:
        """``get_settings()`` 와 ``video_service`` 기본값에 맞춘 스냅샷."""
        s = get_settings()
        from app.services import video_service as vs

        w = width if width is not None else vs.DEFAULT_WIDTH
        h = height if height is not None else vs.DEFAULT_HEIGHT
        nf = num_frames if num_frames is not None else vs.DEFAULT_NUM_FRAMES
        w, h, nf = LTXVideoEngine.normalize_params(w, h, nf)
        steps = getattr(s, "ltx23_num_steps", 8) if "2.3" in (s.ltx2_model_id or "") else vs.DEFAULT_NUM_STEPS
        gs = getattr(s, "ltx23_guidance_scale", 1.0) if "2.3" in (s.ltx2_model_id or "") else vs.DEFAULT_GUIDANCE_SCALE
        return LTXGenerationParams(
            width=w,
            height=h,
            num_frames=nf,
            num_inference_steps=int(steps),
            guidance_scale=float(gs),
            model_id=s.ltx2_model_id,
        )

    @staticmethod
    def note() -> str:
        """문서용 한 줄 설명."""
        return (
            "Inference entrypoint: app.services.video_service.run_image_to_video. "
            "Port: see Settings.port (default 7000)."
        )


def get_engine() -> LTXVideoEngine:
    """싱글턴 대신 상태 없는 빌더만 사용."""
    return LTXVideoEngine()
