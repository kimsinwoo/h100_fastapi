"""
ComfyUI 연동: 로컬 ComfyUI 서버(기본 8188)에 워크플로우를 보내고 결과 이미지를 반환.
backend 루트 기준 app, static, scripts, motions, pipelines 사용.
"""

from __future__ import annotations

import asyncio
import json
import logging
import uuid
from pathlib import Path
from typing import Any

import httpx

from app.core.config import get_settings
from app.utils.file_handler import ensure_generated_dir

logger = logging.getLogger(__name__)


def _unwrap_workflow(workflow: dict[str, Any]) -> dict[str, Any]:
    """
    ComfyUI API 포맷은 두 가지 구조로 올 수 있다:
      - 래핑됨:   {"prompt": {"1": {node}, "2": {node}, ...}}
      - 래핑 안됨: {"1": {node}, "2": {node}, ...}
    항상 노드 dict(래핑 안 된 형태)를 반환한다.
    """
    if (
        len(workflow) == 1
        and "prompt" in workflow
        and isinstance(workflow["prompt"], dict)
    ):
        return workflow["prompt"]
    return workflow


def _load_workflow_json(path: Path) -> dict[str, Any]:
    """워크플로 JSON 로드. 문법 오류 시 위치와 수정 방법을 담은 에러 반환."""
    text = path.read_text(encoding="utf-8")
    try:
        return json.loads(text)
    except json.JSONDecodeError as e:
        snippet = ""
        if e.pos is not None and 0 <= e.pos < len(text):
            start = max(0, e.pos - 40)
            end = min(len(text), e.pos + 40)
            snippet = text[start:end].replace("\n", " ")
        raise RuntimeError(
            f"Invalid workflow JSON in {path.name}: line {e.lineno} column {e.colno} (char {e.pos}). "
            "Re-export from ComfyUI as 'Save (API Format)' and ensure no trailing commas or comments. "
            + (f"Near error: ...{snippet}..." if snippet else "")
        ) from e


def _base() -> str:
    return get_settings().comfyui_base_url.rstrip("/")


async def _post_prompt(prompt: dict[str, Any], client_id: str | None = None) -> dict[str, Any]:
    """POST /prompt → { prompt_id, number } or error."""
    payload: dict[str, Any] = {"prompt": prompt, "prompt_id": str(uuid.uuid4())}
    if client_id:
        payload["client_id"] = client_id
    timeout = get_settings().comfyui_timeout_seconds
    async with httpx.AsyncClient(timeout=timeout) as client:
        r = await client.post(f"{_base()}/prompt", json=payload)
        r.raise_for_status()
        return r.json()


async def _get_history(prompt_id: str) -> dict[str, Any]:
    """GET /history/{prompt_id} → raw response dict. 폴링 측에서 prompt_id 키 여부로 완료 판단."""
    async with httpx.AsyncClient(timeout=30.0) as client:
        r = await client.get(f"{_base()}/history/{prompt_id}")
        r.raise_for_status()
        return r.json()


async def _get_image(filename: str, subfolder: str = "", type_: str = "output") -> bytes:
    """GET /view?filename=...&subfolder=...&type=output → image bytes."""
    params = {"filename": filename, "type": type_}
    if subfolder:
        params["subfolder"] = subfolder
    async with httpx.AsyncClient(timeout=60.0) as client:
        r = await client.get(f"{_base()}/view", params=params)
        r.raise_for_status()
        return r.content


def _prepare_reference_video_for_comfyui(reference_video_path: Path) -> str | None:
    """
    레퍼런스 영상 파일명 반환. ComfyUI가 같은 경로를 읽을 수 있으면 해당 파일명으로 주입.
    COMFYUI_REFERENCE_VIDEO_DIR 미설정 시 레퍼런스 저장 위치(motions_dir)를 사용해 파일명만 반환.
    """
    if not reference_video_path.exists():
        return None
    settings = get_settings()
    ref_dir = getattr(settings, "comfyui_reference_video_dir", None)
    if not ref_dir:
        ref_dir = str(reference_video_path.parent)
        logger.info(
            "COMFYUI_REFERENCE_VIDEO_DIR 미설정. 레퍼런스 경로 기준으로 파일명 전달: dir=%s, name=%s",
            ref_dir, reference_video_path.name,
        )
    dir_path = Path(ref_dir).resolve()
    if not reference_video_path.exists():
        return None
    # 이미 ref_dir 안에 있으면 복사 없이 파일명만 반환 (예: motions_dir 와 동일 경로로 설정한 경우)
    if reference_video_path.parent.resolve() == dir_path:
        return reference_video_path.name
    if not dir_path.is_dir():
        try:
            dir_path.mkdir(parents=True, exist_ok=True)
        except Exception as e:
            logger.warning("ComfyUI 레퍼런스 디렉터리 생성 실패 %s: %s", ref_dir, e)
            return None
    name = f"ref_{uuid.uuid4().hex[:12]}{reference_video_path.suffix or '.mp4'}"
    dest = dir_path / name
    try:
        import shutil
        shutil.copy2(reference_video_path, dest)
        return name
    except Exception as e:
        logger.warning("레퍼런스 영상 복사 실패 %s -> %s: %s", reference_video_path, dest, e)
        return None


async def upload_pose_frames_to_comfyui(
    pose_cache_path: Path | str,
    max_frames: int = 49,
) -> list[str]:
    """
    pose_cache JSON(프레임별 keypoints)을 스켈레톤 이미지로 그린 뒤 ComfyUI에 업로드.
    MotionSequence 형식: { fps, width, height, frames: [{ frame, timestamp, keypoints: [{ joint, x, y }] }] }.
    Returns: ComfyUI에 업로드된 이미지 파일명 리스트.
    """
    path = Path(pose_cache_path)
    if not path.exists():
        logger.warning("포즈 캐시 없음: %s", path)
        return []
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except Exception as e:
        logger.warning("포즈 캐시 로드 실패 %s: %s", path, e)
        return []
    fps = data.get("fps", 8)
    width = int(data.get("width", 768))
    height = int(data.get("height", 512))
    frames_data = data.get("frames", [])[:max_frames]
    if not frames_data:
        logger.warning("포즈 캐시에 프레임 없음: %s", path)
        return []

    # 관절 연결 (joint 이름 기준, MediaPipe Pose 순서)
    CONNECTIONS = [
        ("left_shoulder", "right_shoulder"),
        ("left_shoulder", "left_elbow"), ("left_elbow", "left_wrist"),
        ("right_shoulder", "right_elbow"), ("right_elbow", "right_wrist"),
        ("left_shoulder", "left_hip"), ("right_shoulder", "right_hip"),
        ("left_hip", "right_hip"),
        ("left_hip", "left_knee"), ("left_knee", "left_ankle"),
        ("right_hip", "right_knee"), ("right_knee", "right_ankle"),
    ]

    try:
        from PIL import Image, ImageDraw
    except ImportError:
        logger.warning("PIL 없음. 포즈 프레임 업로드 스킵.")
        return []

    import io
    uploaded_names: list[str] = []
    for i, frame_data in enumerate(frames_data):
        keypoints = frame_data.get("keypoints", [])
        kp_by_joint = {kp.get("joint"): (float(kp.get("x", 0)), float(kp.get("y", 0))) for kp in keypoints if isinstance(kp, dict) and "joint" in kp}
        img = Image.new("RGB", (width, height), (0, 0, 0))
        draw = ImageDraw.Draw(img)
        for j1, j2 in CONNECTIONS:
            if j1 in kp_by_joint and j2 in kp_by_joint:
                x1, y1 = kp_by_joint[j1]
                x2, y2 = kp_by_joint[j2]
                x1, y1 = int(x1 * width), int(y1 * height)
                x2, y2 = int(x2 * width), int(y2 * height)
                draw.line([(x1, y1), (x2, y2)], fill=(255, 255, 255), width=3)
        for _j, (px, py) in kp_by_joint.items():
            px, py = int(px * width), int(py * height)
            draw.ellipse([(px - 4, py - 4), (px + 4, py + 4)], fill=(0, 255, 0))
        buf = io.BytesIO()
        img.save(buf, format="PNG")
        name = f"pose_frame_{i:04d}.png"
        result = await upload_image(buf.getvalue(), name)
        uploaded_names.append(result.get("name", name))
    logger.info("포즈 프레임 %d장 ComfyUI 업로드 완료", len(uploaded_names))
    return uploaded_names


async def upload_image(image_bytes: bytes, filename: str | None = None) -> dict[str, str]:
    """ComfyUI에 이미지 업로드. POST /upload/image → { name, subfolder }."""
    name = filename or f"ltx_input_{uuid.uuid4().hex[:12]}.png"
    if not name.lower().endswith((".png", ".jpg", ".jpeg", ".webp")):
        name += ".png"
    async with httpx.AsyncClient(timeout=30.0) as client:
        r = await client.post(
            f"{_base()}/upload/image",
            files={"image": (name, image_bytes, "image/png")},
            data={"type": "input", "overwrite": "true"},
        )
        r.raise_for_status()
        out = r.json()
        return {"name": out.get("name", name), "subfolder": out.get("subfolder", "")}


def _safe_log_dump(obj: Any, max_len: int = 4000) -> str:
    """디버깅용: dict/list를 JSON 유사 문자열로 변환 (길이 제한)."""
    try:
        out = json.dumps(obj, ensure_ascii=False, indent=2, default=lambda x: str(type(x).__name__))
        return out[:max_len] + ("..." if len(out) > max_len else "")
    except Exception:
        return repr(obj)[:max_len]


def _extract_first_video_from_history(history: dict[str, Any]) -> tuple[str, str, str] | None:
    """history에서 첫 비디오 출력 (SaveVideo / VHS_VideoCombine 등). (filename, subfolder, type)."""
    if "outputs" in history:
        outputs = history["outputs"]
    else:
        for k, v in history.items():
            if isinstance(v, dict) and "outputs" in v:
                outputs = v["outputs"]
                break
        else:
            return None
    if not isinstance(outputs, dict):
        return None
    for _node_id, node_out in outputs.items():
        if not isinstance(node_out, dict):
            continue
        # 노드 최상단에 filename 있는 경우 (일부 워크플로)
        top_fn = node_out.get("filename") or node_out.get("file_name")
        if isinstance(top_fn, str) and top_fn.strip().endswith((".mp4", ".webm", ".gif", ".mov")):
            return (top_fn.strip(), node_out.get("subfolder", ""), node_out.get("type", "output"))
        # SaveVideo 등: videos, gifs, images, animated 순으로 확인. animated가 [true]처럼 비파일이면 스킵.
        for key in ("videos", "gifs", "images", "animations", "animated", "video"):
            raw = node_out.get(key)
            if raw is None:
                continue
            videos = raw if isinstance(raw, list) else [raw]
            if not videos:
                continue
            first = videos[0]
            if isinstance(first, dict):
                inner_val = first.get("value") or first.get("output") or first.get("result") or {}
                inner_val = inner_val if isinstance(inner_val, dict) else {}
                fn = (
                    first.get("filename")
                    or first.get("file_name")
                    or first.get("name")
                    or first.get("path")
                    or inner_val.get("filename")
                    or inner_val.get("file_name")
                    or inner_val.get("name")
                )
                if not fn:
                    for _k, v in first.items():
                        if isinstance(v, str) and v.strip().endswith((".mp4", ".webm", ".gif", ".mov")):
                            fn = v.strip()
                            break
                if fn:
                    inner = inner_val if inner_val else first
                    return (fn, inner.get("subfolder", ""), inner.get("type", "output"))
            elif isinstance(first, str) and first.strip().endswith((".mp4", ".webm", ".gif", ".mov")):
                return (first.strip(), "", "output")
            elif isinstance(first, (list, tuple)) and len(first) >= 1:
                return (str(first[0]), str(first[1]) if len(first) > 1 else "", str(first[2]) if len(first) > 2 else "output")
    return None


async def _get_video_bytes(filename: str, subfolder: str = "", type_: str = "output") -> bytes:
    """ComfyUI 출력 비디오 bytes. comfyui_output_dir 설정 시 로컬 파일에서 읽고, 없으면 /view 시도."""
    settings = get_settings()
    out_dir = getattr(settings, "comfyui_output_dir", None)
    if out_dir:
        path = Path(out_dir)
        if subfolder:
            path = path / subfolder
        path = path / filename
        if path.exists():
            return path.read_bytes()
    async with httpx.AsyncClient(timeout=120.0) as client:
        params = {"filename": filename, "type": type_}
        if subfolder:
            params["subfolder"] = subfolder
        r = await client.get(f"{_base()}/view", params=params)
        r.raise_for_status()
        return r.content


def _extract_first_image_from_history(history: dict[str, Any]) -> tuple[str, str, str] | None:
    """history[prompt_id] 또는 history 자체에서 첫 SaveImage 출력 (filename, subfolder, type)."""
    # 구조: history = { prompt_id: { outputs: { node_id: { images: [ {filename, subfolder, type} ] } } } }
    if "outputs" in history:
        outputs = history["outputs"]
    else:
        # 단일 prompt_id 키가 있으면 그걸 사용
        for k, v in history.items():
            if isinstance(v, dict) and "outputs" in v:
                outputs = v["outputs"]
                break
        else:
            return None
    for _node_id, node_out in outputs.items():
        if not isinstance(node_out, dict):
            continue
        images = node_out.get("images") or node_out.get("gifs")
        if not images:
            continue
        img = images[0] if isinstance(images[0], dict) else {}
        fn = img.get("filename")
        if fn:
            return (
                fn,
                img.get("subfolder", ""),
                img.get("type", "output"),
            )
    return None


async def run_workflow_and_get_image(
    workflow: dict[str, Any],
    client_id: str | None = None,
    poll_interval: float = 0.5,
    max_wait: float | None = None,
) -> bytes:
    """
    ComfyUI에 워크플로우 제출 → 완료까지 폴링 → 첫 출력 이미지 bytes 반환.
    """
    settings = get_settings()
    if not settings.comfyui_enabled:
        raise RuntimeError("ComfyUI is disabled (COMFYUI_ENABLED=false)")
    max_wait = max_wait or settings.comfyui_timeout_seconds

    out = await _post_prompt(workflow, client_id=client_id)
    if "error" in out:
        raise RuntimeError(f"ComfyUI prompt error: {out['error']}")
    prompt_id = out.get("prompt_id")
    if not prompt_id and "number" in out:
        prompt_id = str(out.get("number", ""))
    if not prompt_id:
        raise RuntimeError("ComfyUI did not return prompt_id")

    start = asyncio.get_event_loop().time()
    while (asyncio.get_event_loop().time() - start) < max_wait:
        history = await _get_history(prompt_id)
        # 완료: 해당 prompt_id 에 대한 실행 결과가 있음
        if prompt_id in history:
            info = history[prompt_id]
            if isinstance(info, dict):
                first = _extract_first_image_from_history(info)
                if first:
                    filename, subfolder, type_ = first
                    return await _get_image(filename, subfolder, type_)
        # 아직 실행 중이면 outputs 가 비어있을 수 있음
        await asyncio.sleep(poll_interval)

    raise TimeoutError(f"ComfyUI workflow {prompt_id} did not finish within {max_wait}s")


async def run_workflow_save_to_generated(
    workflow: dict[str, Any],
    output_name: str | None = None,
) -> Path:
    """워크플로우 실행 후 이미지를 static/generated 에 저장하고 경로 반환."""
    ensure_generated_dir()
    settings = get_settings()
    generated = settings.generated_dir
    name = output_name or f"comfyui_{uuid.uuid4().hex[:12]}.png"
    if not name.endswith(".png"):
        name += ".png"
    path = generated / name

    data = await run_workflow_and_get_image(workflow)
    path.write_bytes(data)
    logger.info("ComfyUI output saved: %s", path)
    return path


async def health_check() -> bool:
    """ComfyUI 서버 연결 가능 여부."""
    try:
        async with httpx.AsyncClient(timeout=5.0) as client:
            r = await client.get(f"{_base()}/system_stats")
            return r.status_code == 200
    except Exception as e:
        logger.debug("ComfyUI health check failed: %s", e)
        return False


def _load_workflow_graph(
    workflow_name: str,
    pipelines_dir: Path,
    prefer_ref: bool = False,
) -> dict[str, Any]:
    """
    워크플로우 로드. 우선순위:
      1) prefer_ref 시 pipelines/{name}_ref.json
      2) pipelines/{name}.json
      3) pipelines/comfyui_ltx23_workflow.json
    반환: ComfyUI API용 prompt 맵 (node_id -> node_spec).
    """
    candidates: list[Path] = []
    if prefer_ref:
        candidates.append(pipelines_dir / f"{workflow_name}_ref.json")
    candidates.append(pipelines_dir / f"{workflow_name}.json")
    candidates.append(pipelines_dir / "comfyui_ltx23_workflow.json")

    for path in candidates:
        if path.exists():
            raw = _load_workflow_json(path)
            graph = raw.get("prompt") if isinstance(raw.get("prompt"), dict) else raw
            if isinstance(graph, dict):
                logger.info("워크플로우 로드: %s", path.name)
                return graph
    raise FileNotFoundError(
        f"워크플로우 파일을 찾을 수 없습니다. 탐색: {[str(c) for c in candidates]}. "
        "scripts/patch_workflow.py 를 실행한 뒤 다시 시도하세요."
    )


def inject_reference_video(workflow: dict[str, Any], video_filename: str) -> None:
    """
    워크플로우의 VHS_LoadVideo 계열 노드에 댄스 영상 파일명을 주입 (in-place).
    래핑된 구조({"prompt": {...}})여도 _unwrap_workflow로 노드만 순회.
    """
    nodes = _unwrap_workflow(workflow)
    LOAD_VIDEO_TYPES = frozenset(("vhs_loadvideo", "vhs_loadvideopath", "loadvideo"))
    for node_id, node in nodes.items():
        if not isinstance(node, dict):
            continue
        ct = (node.get("class_type") or "").strip()
        ct_lower = ct.lower().replace(" ", "")
        is_video_node = (
            ct_lower in LOAD_VIDEO_TYPES
            or ("load" in ct_lower and "video" in ct_lower)
        )
        if not is_video_node:
            continue
        inputs = node.get("inputs") or {}
        for key in ("video", "file_path", "path", "filename"):
            if key in inputs:
                node["inputs"][key] = video_filename
                logger.info(
                    "[inject_reference_video] 완료: 노드 %s (%s).%s ← %s",
                    node_id, ct, key, video_filename,
                )
                return
        node["inputs"] = {**inputs, "video": video_filename}
        logger.info(
            "[inject_reference_video] 강제 주입: 노드 %s (%s) ← %s",
            node_id, ct, video_filename,
        )
        return
    logger.warning(
        "워크플로우에 VHS_LoadVideo 노드 없음. backend에서 python scripts/patch_workflow.py 를 실행하세요."
    )


def _inject_ltx23_workflow_inputs(
    workflow: dict[str, Any],
    image_name: str,
    prompt_text: str,
    negative_prompt_text: str = "",
    reference_video_filename: str | None = None,
) -> dict[str, Any]:
    """워크플로에 이미지·프롬프트(positive/negative)·(선택)레퍼런스 주입. 래핑된 구조도 _unwrap_workflow로 노드만 순회."""
    import copy
    wf = copy.deepcopy(workflow)
    nodes = _unwrap_workflow(wf)
    # 1) PrimitiveStringMultiline "Prompt" → value 주입
    for nid, node in nodes.items():
        if not isinstance(node, dict):
            continue
        ct = (node.get("class_type") or "").lower()
        meta = (node.get("_meta") or {}).get("title") or ""
        inputs = node.get("inputs") or {}
        if "primitivestringmultiline" in ct and "prompt" in meta.lower() and "value" in inputs:
            node["inputs"] = {**inputs, "value": prompt_text}
            logger.info("[inject] 프롬프트(PrimitiveStringMultiline): 노드 %s value 길이=%d", nid, len(prompt_text))
            break
    # 2) CLIPTextEncode: positive/negative 주입
    clip_text_nodes = [
        (nid, node) for nid, node in nodes.items()
        if isinstance(node, dict)
        and "clip" in (node.get("class_type") or "").lower()
        and "text" in (node.get("class_type") or "").lower()
    ]
    clip_text_nodes.sort(key=lambda x: x[0])
    for i, (nid, node) in enumerate(clip_text_nodes):
        inputs = node.get("inputs") or {}
        text = prompt_text if i == 0 else (negative_prompt_text or "")
        node["inputs"] = {**inputs, "text": text}
    if clip_text_nodes:
        logger.info(
            "ComfyUI CLIP 주입: positive=%s(len=%d), negative=%s(len=%d)",
            clip_text_nodes[0][0], len(prompt_text),
            clip_text_nodes[1][0] if len(clip_text_nodes) > 1 else "-", len(negative_prompt_text),
        )
    # 3) 이미지 로더 + 레퍼런스 비디오 로더
    REF_VIDEO_NODE_TYPES = frozenset(("vhs_loadvideo", "vhs_loadvideopath", "loadvideo"))

    def _is_video_loader_node(class_type: str) -> bool:
        c = (class_type or "").lower().replace(" ", "")
        return c in REF_VIDEO_NODE_TYPES or ("load" in c and "video" in c)

    ref_injected = False
    for nid, node in nodes.items():
        if not isinstance(node, dict):
            continue
        ct = node.get("class_type") or ""
        ct_lower = ct.lower()
        inputs = node.get("inputs") or {}
        if "loadimage" in ct_lower or "load image" in ct_lower:
            node["inputs"] = {**inputs, "image": image_name}
            logger.info("[inject] 이미지 주입: 노드 %s (%s) ← %s", nid, ct, image_name)
        if reference_video_filename and _is_video_loader_node(ct):
            for key in ("video", "file_path", "path", "filename", "input"):
                if key in inputs:
                    node["inputs"] = {**inputs, key: reference_video_filename}
                    logger.info("[inject] 댄스 영상 주입: 노드 %s (%s).%s ← %s", nid, ct, key, reference_video_filename)
                    ref_injected = True
                    break
            if not ref_injected:
                node["inputs"] = {**inputs, "video": reference_video_filename}
                logger.info("[inject] 댄스 영상 강제 주입: 노드 %s (%s) ← %s", nid, ct, reference_video_filename)
                ref_injected = True
            break
    if reference_video_filename and not ref_injected:
        logger.warning(
            "레퍼런스 파일명=%s 이지만 워크플로에 레퍼런스용 비디오 입력 노드가 없어 주입되지 않음. "
            "ComfyUI에서 VHS_LoadVideo 노드를 추가하고 API 포맷으로 저장하세요.",
            reference_video_filename,
        )
    return wf


async def run_ltx23_image_to_video(
    image_bytes: bytes,
    prompt: str,
    negative_prompt: str = "",
    width: int = 768,
    height: int = 512,
    num_frames: int = 121,
    frame_rate: float = 24.0,
    num_inference_steps: int = 8,
    guidance_scale: float = 1.0,
    seed: int | None = None,
    reference_video_path: Path | None = None,
) -> bytes:
    """
    LTX-2.3 이미지→비디오를 ComfyUI(LTXVideo 노드)로 실행.
    모델·파이프라인은 ComfyUI/models 및 pipelines 참조. 워크플로는 pipelines/<comfyui_ltx23_workflow>.json.
    """
    settings = get_settings()
    if not settings.comfyui_enabled:
        raise RuntimeError("ComfyUI is disabled (COMFYUI_ENABLED=false)")

    workflow_name = getattr(settings, "comfyui_ltx23_workflow", "ltx23_i2v") or "ltx23_i2v"
    prefer_ref = bool(reference_video_path and reference_video_path.exists())
    try:
        graph = _load_workflow_graph(workflow_name, settings.pipelines_dir, prefer_ref=prefer_ref)
    except FileNotFoundError as e:
        raise RuntimeError(str(e)) from e

    base_url = settings.comfyui_base_url
    try:
        uploaded = await upload_image(image_bytes)
        image_name = uploaded.get("name", "image.png")
        ref_filename: str | None = None
        if reference_video_path and reference_video_path.exists():
            ref_filename = _prepare_reference_video_for_comfyui(reference_video_path)
            if ref_filename:
                logger.info("레퍼런스 영상을 ComfyUI용으로 준비함: %s", ref_filename)
                inject_reference_video(graph, ref_filename)  # 반드시 먼저 주입, 노드 없으면 여기서 에러
            elif getattr(get_settings(), "comfyui_reference_video_dir", None):
                logger.warning("레퍼런스 영상 복사 실패. COMFYUI_REFERENCE_VIDEO_DIR 경로 확인.")
            else:
                logger.info(
                    "레퍼런스 영상이 ComfyUI에 전달되지 않음. "
                    "동일 동작 반영을 위해 .env에 COMFYUI_REFERENCE_VIDEO_DIR=ComfyUI의 input 폴더 경로 를 설정하고, "
                    "워크플로에 레퍼런스용 LoadVideo 노드가 있어야 합니다."
                )
        prompt_graph = _inject_ltx23_workflow_inputs(
            graph, image_name, prompt,
            negative_prompt_text=negative_prompt or "",
            reference_video_filename=ref_filename,
        )
        # ComfyUI API는 노드 dict만 받음. 래핑된 구조면 언래핑 후 전송
        prompt_to_send = _unwrap_workflow(prompt_graph)
        out = await _post_prompt(prompt_to_send)
        if "error" in out:
            raise RuntimeError(f"ComfyUI LTX-2.3 prompt error: {out['error']}")
        prompt_id = out.get("prompt_id") or str(out.get("number", ""))
        if not prompt_id:
            raise RuntimeError("ComfyUI did not return prompt_id")

        max_wait = getattr(settings, "comfyui_video_timeout_seconds", None) or settings.comfyui_timeout_seconds
        poll_interval = 2.0  # 0.5→2초: 로그·부하 감소, 영상 워크플로는 완료까지 시간 걸림
        start = asyncio.get_event_loop().time()
        last_log_at = 0.0
        full_dump_logged = False  # 비디오 못 찾을 때 전체 구조는 한 번만 출력
        while (asyncio.get_event_loop().time() - start) < max_wait:
            history = await _get_history(prompt_id)
            # ComfyUI 응답 두 가지: 1) { prompt_id: { outputs:... } }  2) 단일 항목이면 { outputs:... } 만 옴
            if prompt_id in history and isinstance(history[prompt_id], dict):
                info = history[prompt_id]
            elif "outputs" in history and isinstance(history.get("outputs"), dict):
                info = history
            else:
                info = None
            if info is not None:
                first_video = _extract_first_video_from_history(info)
                if first_video:
                    filename, subfolder, type_ = first_video
                    return await _get_video_bytes(filename, subfolder, type_)
                # 완료됐는데 비디오 못 찾은 경우: 전체 구조 한 번만 상세 출력 (이유 파악용)
                if not full_dump_logged:
                    full_dump_logged = True
                    outputs = info.get("outputs") if isinstance(info.get("outputs"), dict) else {}
                    logger.info(
                        "[ComfyUI history 전체] prompt_id=%s, API 응답 최상위 키(history)=%s, info 최상위 키=%s",
                        prompt_id[:8],
                        list(history.keys()),
                        list(info.keys()),
                    )
                    logger.info("[ComfyUI history] raw response (info) =\n%s", _safe_log_dump(info, max_len=8000))
                    for nid, node_out in outputs.items():
                        if not isinstance(node_out, dict):
                            logger.info("[ComfyUI 노드 %s] 타입=%s", nid, type(node_out).__name__)
                            continue
                        logger.info(
                            "[ComfyUI 노드 %s] 키=%s",
                            nid,
                            list(node_out.keys()),
                        )
                        for key in ("images", "animated", "videos", "gifs", "animations"):
                            val = node_out.get(key)
                            if val is None:
                                continue
                            if isinstance(val, list):
                                logger.info(
                                    "[ComfyUI 노드 %s.%s] len=%s, 첫 항목 타입=%s, 첫 항목 내용=\n%s",
                                    nid,
                                    key,
                                    len(val),
                                    type(val[0]).__name__ if val else "N/A",
                                    _safe_log_dump(val[0], max_len=3000) if val else "[]",
                                )
                                if len(val) > 1:
                                    logger.info(
                                        "[ComfyUI 노드 %s.%s] 두 번째 항목=\n%s",
                                        nid,
                                        key,
                                        _safe_log_dump(val[1], max_len=1500) if len(val) > 1 else "N/A",
                                    )
                            else:
                                logger.info(
                                    "[ComfyUI 노드 %s.%s] 타입=%s, 값=\n%s",
                                    nid,
                                    key,
                                    type(val).__name__,
                                    _safe_log_dump(val, max_len=2000),
                                )
            elapsed = asyncio.get_event_loop().time() - start
            if elapsed - last_log_at >= 30.0:
                logger.info(
                    "ComfyUI LTX-2.3 대기 중 prompt_id=%s 경과 %.0fs (최대 %.0fs)",
                    prompt_id[:8], elapsed, max_wait,
                )
                last_log_at = elapsed
            await asyncio.sleep(poll_interval)

        raise TimeoutError(
            f"ComfyUI LTX-2.3 workflow {prompt_id} did not finish within {max_wait}s. "
            "영상 생성이 제한 시간 내에 완료되지 않았습니다. ComfyUI 서버 상태를 확인하거나 "
            "COMFYUI_VIDEO_TIMEOUT_SECONDS를 늘린 후 다시 시도해 주세요."
        )
    except (httpx.ConnectError, httpx.ConnectTimeout) as e:
        raise RuntimeError(
            f"ComfyUI server unreachable at {base_url}. "
            "Ensure ComfyUI is running and COMFYUI_BASE_URL is correct (e.g. http://comfyui:8188 if in another pod/container)."
        ) from e
    except httpx.HTTPStatusError as e:
        body = ""
        if e.response is not None:
            try:
                body = e.response.text[:500] if e.response.text else ""
            except Exception:
                pass
        # VHS_LoadVideo 등 missing_node_type → Video Helper Suite 미설치 안내
        if e.response and e.response.status_code == 400 and body:
            try:
                import json as _json
                data = _json.loads(e.response.text) if e.response.text else {}
                err = data.get("error") or {}
                if err.get("type") == "missing_node_type":
                    class_type = (err.get("extra_info") or {}).get("class_type") or ""
                    if "VHS_LoadVideo" in class_type or "LoadVideo" in (err.get("message") or ""):
                        raise RuntimeError(
                            "ComfyUI에 Video Helper Suite가 설치되어 있지 않습니다. "
                            "레퍼런스 댄스 영상을 사용하려면 다음을 실행한 뒤 ComfyUI를 재시작하세요:\n"
                            "  cd ComfyUI/custom_nodes\n"
                            "  git clone https://github.com/Kosinkadink/ComfyUI-VideoHelperSuite\n"
                            "  pip install -r ComfyUI-VideoHelperSuite/requirements.txt"
                        ) from e
            except RuntimeError:
                raise
            except Exception:
                pass
        msg = f"ComfyUI returned {e.response.status_code if e.response else 'error'} for /prompt."
        if body:
            msg += f" Response: {body}"
        raise RuntimeError(msg) from e
