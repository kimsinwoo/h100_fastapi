"""
채팅방 저장·조회. user_id 기준 완전 분리 (로그인 없이 브라우저 단위 격리).
구조: chat_rooms_dir / {user_id} / {room_id}.json
"""

from __future__ import annotations

import json
import uuid
from pathlib import Path
from typing import Any

from app.core.config import get_settings


def _rooms_dir(user_id: str) -> Path:
    """user_id별 채팅방 디렉터리."""
    base = Path(get_settings().chat_rooms_dir)
    d = base / user_id.strip()
    d.mkdir(parents=True, exist_ok=True)
    return d


def _room_path(room_id: str, user_id: str) -> Path:
    return _rooms_dir(user_id) / f"{room_id}.json"


def _read_room(room_id: str, user_id: str) -> dict[str, Any] | None:
    p = _room_path(room_id, user_id)
    if not p.is_file():
        return None
    try:
        data = json.loads(p.read_text(encoding="utf-8"))
        data.setdefault("id", room_id)
        if data.get("user_id") != user_id:
            return None
        return data
    except Exception:
        return None


def _write_room(data: dict[str, Any], user_id: str) -> None:
    room_id = data.get("id")
    if not room_id:
        return
    data["user_id"] = user_id
    p = _room_path(room_id, user_id)
    p.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")


def list_rooms(user_id: str) -> list[dict[str, Any]]:
    """user_id의 채팅방 목록 (최신 수정순)."""
    if not user_id or not user_id.strip():
        return []
    dir_path = _rooms_dir(user_id)
    rooms = []
    for f in dir_path.glob("*.json"):
        room_id = f.stem
        data = _read_room(room_id, user_id)
        if data is None:
            continue
        rooms.append({
            "id": data.get("id", room_id),
            "title": data.get("title") or "새 대화",
            "updated_at": data.get("updated_at", ""),
        })
    rooms.sort(key=lambda r: r.get("updated_at") or "", reverse=True)
    return rooms


def get_room(room_id: str, user_id: str) -> dict[str, Any] | None:
    """채팅방 한 건 조회. user_id 소유만 반환."""
    if not user_id or not user_id.strip():
        return None
    data = _read_room(room_id, user_id)
    if data is None:
        return None
    return {k: v for k, v in data.items() if k != "user_id"}


def create_room(user_id: str, title: str | None = None) -> dict[str, Any]:
    """user_id 소유 채팅방 생성."""
    if not user_id or not user_id.strip():
        raise ValueError("user_id required")
    room_id = uuid.uuid4().hex
    import datetime
    now = datetime.datetime.utcnow().isoformat() + "Z"
    data = {
        "id": room_id,
        "user_id": user_id,
        "title": title or "새 대화",
        "messages": [],
        "created_at": now,
        "updated_at": now,
    }
    _write_room(data, user_id)
    return {k: v for k, v in data.items() if k != "user_id"}


def add_message(room_id: str, user_id: str, role: str, content: str) -> dict[str, Any] | None:
    """메시지 추가. 해당 room이 user_id 소유일 때만."""
    if not user_id or not user_id.strip():
        return None
    data = _read_room(room_id, user_id)
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
    _write_room(data, user_id)
    return {k: v for k, v in data.items() if k != "user_id"}


def delete_room(room_id: str, user_id: str) -> bool:
    """user_id 소유 채팅방만 삭제."""
    if not user_id or not user_id.strip():
        return False
    p = _room_path(room_id, user_id)
    if not p.is_file():
        return False
    try:
        p.unlink()
        return True
    except Exception:
        return False
