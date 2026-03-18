"""
앱 공통 예외. HTTP 상태 코드와 메시지 포함.
"""

from __future__ import annotations


class DogDanceException(Exception):
    """강아지 댄스/영상 생성 관련 예외 베이스."""

    def __init__(self, message: str, status_code: int = 500):
        super().__init__(message)
        self.message = message
        self.status_code = status_code


class DanceNotFoundError(DogDanceException):
    """지정한 dance_video_id가 dance_videos/ 폴더에 없음."""

    def __init__(self, dance_id: str):
        super().__init__(
            message=f"'{dance_id}' 댄스 영상을 찾을 수 없습니다. GET /api/dance/list 로 사용 가능한 목록을 확인하세요.",
            status_code=404,
        )
        self.dance_id = dance_id


class DanceVideoDirNotFoundError(DogDanceException):
    """dance_videos/ 폴더 자체가 존재하지 않음."""

    def __init__(self, path: str):
        super().__init__(
            message=f"댄스 영상 폴더({path})가 존재하지 않습니다. 폴더를 생성하고 mp4 파일을 넣어주세요.",
            status_code=500,
        )


class EmptyDanceLibraryError(DogDanceException):
    """dance_videos/ 폴더에 파일이 없음."""

    def __init__(self, path: str):
        super().__init__(
            message=f"댄스 영상 폴더({path})가 비어있습니다. mp4, mov, avi 파일을 추가 후 POST /api/dance/refresh 를 호출하세요.",
            status_code=404,
        )


class WorkflowVideoNodeNotFoundError(DogDanceException):
    """
    워크플로우에 VHS_LoadVideo 노드가 없을 때 발생.
    ComfyUI UI에서 VHS_LoadVideo 노드 추가 후 API 포맷 export 미완료를 의미.
    """

    def __init__(self, detail: str = ""):
        super().__init__(
            message=(
                "워크플로우에 댄스 영상 입력 노드(VHS_LoadVideo)가 없습니다. "
                "ComfyUI UI에서 VHS_LoadVideo 노드를 추가하고 "
                "API 포맷으로 다시 export한 뒤 ltx23_i2v.json을 교체하세요. "
                f"{detail}"
            ).strip(),
            status_code=500,
        )
