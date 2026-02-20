"""
채팅방 저장·조회. 파일 기반(JSON, 채팅방당 1파일).
"""

from __future__ import annotations

import json
import uuid
from pathlib import Path
from typing import Any

from app.core.config import get_settings


def _rooms_dir() -> Path:
    d = Path(get_settings().chat_rooms_dir)
    d.mkdir(parents=True, exist_ok=True)
    return d


def _room_path(room_id: str) -> Path:
    return _rooms_dir() / f"{room_id}.json"


def _read_room(room_id: str) -> dict[str, Any] | None:
    p = _room_path(room_id)
    if not p.is_file():
        return None
    try:
        data = json.loads(p.read_text(encoding="utf-8"))
        data.setdefault("id", room_id)
        return data
    except Exception:
        return None


def _write_room(data: dict[str, Any]) -> None:
    room_id = data.get("id")
    if not room_id:
        return
    p = _room_path(room_id)
    p.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")


def list_rooms() -> list[dict[str, Any]]:
    """채팅방 목록 (최신 수정순)."""
    dir_path = _rooms_dir()
    rooms = []
    for f in dir_path.glob("*.json"):
        room_id = f.stem
        data = _read_room(room_id)
        if data is None:
            continue
        rooms.append({
            "id": data.get("id", room_id),
            "title": data.get("title") or "새 대화",
            "updated_at": data.get("updated_at", ""),
        })
    rooms.sort(key=lambda r: r.get("updated_at") or "", reverse=True)
    return rooms


def get_room(room_id: str) -> dict[str, Any] | None:
    """채팅방 한 건 조회 (메시지 포함)."""
    return _read_room(room_id)


def create_room(title: str | None = None) -> dict[str, Any]:
    """채팅방 생성. 반환: { id, title, messages, created_at, updated_at }."""
    room_id = uuid.uuid4().hex
    import datetime
    now = datetime.datetime.utcnow().isoformat() + "Z"
    data = {
        "id": room_id,
        "title": title or "새 대화",
        "messages": [],
        "created_at": now,
        "updated_at": now,
    }
    _write_room(data)
    return data


def add_message(room_id: str, role: str, content: str) -> dict[str, Any] | None:
    """메시지 추가. 제목이 '새 대화'이고 user 메시지면 첫 문장으로 제목 설정."""
    data = _read_room(room_id)
    if data is None:
        return None
    import datetime
    now = datetime.datetime.utcnow().isoformat() + "Z"
    data["messages"] = data.get("messages") or []
    data["messages"].append({"role": role, "content": content})
    data["updated_at"] = now
    if (data.get("title") or "").strip() in ("", "새 대화") and role == "user":
        title = (content.strip() or "새 대화")[:50]
        if title:
            data["title"] = title
    _write_room(data)
    return data


def delete_room(room_id: str) -> bool:
    """채팅방 삭제."""
    p = _room_path(room_id)
    if not p.is_file():
        return False
    try:
        p.unlink()
        return True
    except Exception:
        return False
