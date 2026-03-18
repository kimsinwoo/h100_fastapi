"""
댄스 라이브러리: dance_videos/ (또는 motions/) 폴더를 스캔해 사전 등록된 댄스 영상 목록을 관리합니다.
입력은 서버 파일시스템에 등록된 영상 ID로만 선택하고, 업로드는 받지 않습니다.
"""

from __future__ import annotations

import asyncio
import json
import logging
import re
import subprocess
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

logger = logging.getLogger(__name__)


@dataclass
class DanceVideo:
    """사전 등록된 댄스 영상 한 건 메타데이터."""

    id: str
    filename: str
    path: str
    display_name: str
    duration_seconds: float
    fps: float
    width: int
    height: int
    frame_count: int
    file_size_mb: float


class DanceLibrary:
    """dance_videos/ 폴더 스캔 및 캐시. 댄스 영상은 업로드 없이 ID로만 선택."""

    SUPPORTED_EXTENSIONS = {".mp4", ".mov", ".avi", ".webm"}

    def __init__(self, dance_dir: str | Path):
        self.dance_dir = Path(dance_dir)
        self._cache: dict[str, DanceVideo] = {}

    async def scan(self) -> list[DanceVideo]:
        """폴더를 스캔해 캐시를 갱신한 뒤 전체 목록 반환."""
        self._cache.clear()
        if not self.dance_dir.exists():
            self.dance_dir.mkdir(parents=True, exist_ok=True)
            logger.warning("댄스 영상 폴더가 없어 생성함: %s", self.dance_dir)
            return []
        videos = []
        for file in sorted(self.dance_dir.iterdir()):
            if file.suffix.lower() in self.SUPPORTED_EXTENSIONS and file.is_file():
                info = await self._extract_info(file)
                if info:
                    self._cache[info.id] = info
                    videos.append(info)
        return videos

    async def get(self, dance_id: str) -> Optional[DanceVideo]:
        """ID로 댄스 영상 조회. 캐시가 비어 있으면 먼저 스캔."""
        if not self._cache:
            await self.scan()
        return self._cache.get(dance_id)

    async def list_all(self) -> list[DanceVideo]:
        """전체 댄스 목록 반환 (캐시 사용)."""
        if not self._cache:
            await self.scan()
        return list(self._cache.values())

    @staticmethod
    def _make_id(filename: str) -> str:
        """파일명 → slug ID. 'robot dance (2).mp4' → 'robot_dance_2'."""
        stem = Path(filename).stem
        slug = re.sub(r"[^\w\s-]", "", stem.lower())
        slug = re.sub(r"[\s-]+", "_", slug).strip("_")
        return slug or "video"

    @staticmethod
    def _make_display_name(filename: str) -> str:
        """파일명 → 표시 이름. 'robot_dance.mp4' → 'Robot Dance'."""
        stem = Path(filename).stem
        return stem.replace("_", " ").replace("-", " ").title()

    async def _extract_info(self, file: Path) -> Optional[DanceVideo]:
        """ffprobe로 영상 메타데이터 추출 (비동기에서 실행)."""
        try:
            result = await asyncio.get_event_loop().run_in_executor(
                None,
                lambda: subprocess.run(
                    [
                        "ffprobe",
                        "-v",
                        "quiet",
                        "-print_format",
                        "json",
                        "-show_format",
                        "-show_streams",
                        str(file),
                    ],
                    capture_output=True,
                    text=True,
                    timeout=10,
                ),
            )
            if result.returncode != 0 or not result.stdout:
                logger.warning("ffprobe 실패: %s — %s", file.name, result.stderr)
                return None
            probe = json.loads(result.stdout)
            streams = probe.get("streams") or []
            video_stream = next((s for s in streams if s.get("codec_type") == "video"), None)
            if not video_stream:
                return None
            fmt = probe.get("format") or {}
            duration = float(fmt.get("duration", 0))
            fps_raw = video_stream.get("r_frame_rate", "8/1")
            if "/" in str(fps_raw):
                num, den = map(int, str(fps_raw).split("/"))
                fps = num / den if den else 8.0
            else:
                fps = float(fps_raw) if fps_raw else 8.0
            nb_frames = video_stream.get("nb_frames")
            frame_count = int(nb_frames) if nb_frames is not None else int(duration * fps)
            size = int(fmt.get("size", 0))
            return DanceVideo(
                id=self._make_id(file.name),
                filename=file.name,
                path=str(file.absolute()),
                display_name=self._make_display_name(file.name),
                duration_seconds=round(duration, 2),
                fps=round(fps, 2),
                width=int(video_stream.get("width", 0)),
                height=int(video_stream.get("height", 0)),
                frame_count=frame_count,
                file_size_mb=round(size / 1024 / 1024, 2),
            )
        except FileNotFoundError:
            logger.warning("ffprobe를 찾을 수 없습니다. ffmpeg 설치 후 다시 시도하세요.")
            return None
        except Exception as e:
            logger.warning("댄스 영상 메타데이터 추출 실패: %s — %s", file.name, e)
            return None
