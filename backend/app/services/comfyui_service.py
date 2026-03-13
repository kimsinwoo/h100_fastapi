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


async def _get_object_info() -> dict[str, Any]:
    """GET /object_info → node class type별 input/output 스키마."""
    async with httpx.AsyncClient(timeout=30.0) as client:
        r = await client.get(f"{_base()}/object_info")
        r.raise_for_status()
        return r.json()


def _ui_workflow_to_prompt(raw: dict[str, Any], object_info: dict[str, Any]) -> dict[str, Any]:
    """ComfyUI UI 저장 형식(nodes + links)을 /prompt API용 prompt 맵으로 변환."""
    nodes_list = raw.get("nodes")
    links_list = raw.get("links") or []
    if not isinstance(nodes_list, list) or len(nodes_list) == 0:
        return {}

    # link_id -> (source_node_id, source_slot)
    link_to_src: dict[int, tuple[int, int]] = {}
    for link in links_list:
        if not isinstance(link, (list, tuple)) or len(link) < 5:
            continue
        link_id = link[0]
        src_id, src_slot = int(link[1]), int(link[2])
        link_to_src[link_id] = (src_id, src_slot)

    prompt: dict[str, dict[str, Any]] = {}
    for node in nodes_list:
        if not isinstance(node, dict):
            continue
        nid = node.get("id")
        ntype = node.get("type")
        if nid is None or not ntype:
            continue
        nid_str = str(int(nid))
        inputs: dict[str, Any] = {}
        # UI에서 inputs: [ [link_id, node_id, slot, type], ... ] (슬롯 순서)
        node_inputs = node.get("inputs") or []
        widgets_values = node.get("widgets_values") or []
        if not isinstance(node_inputs, list):
            node_inputs = []

        # object_info: input_order = { "required": ["name1", ...], "optional": ["opt1", ...] }
        info = object_info.get(ntype, {}) or {}
        io = info.get("input_order") if isinstance(info.get("input_order"), dict) else {}
        input_order: list[str] = list(io.get("required") or []) + list(io.get("optional") or [])

        # UI inputs: 리스트일 수 있음. [ [link_id, node_id, slot, type], ... ] 또는 [ { "name", "link", ... }, ... ]
        widget_idx = 0
        for slot_i, conn in enumerate(node_inputs):
            input_name = input_order[slot_i] if slot_i < len(input_order) else str(slot_i)
            link_id = None
            if isinstance(conn, (list, tuple)) and len(conn) >= 1:
                link_id = conn[0]
            elif isinstance(conn, dict) and "link" in conn and conn["link"] is not None:
                link_id = conn["link"]
                if "name" in conn:
                    input_name = conn["name"]
            if link_id is not None and link_id in link_to_src:
                src_id, src_slot = link_to_src[link_id]
                inputs[input_name] = [str(src_id), src_slot]
            else:
                if widget_idx < len(widgets_values):
                    inputs[input_name] = widgets_values[widget_idx]
                    widget_idx += 1
        # 남은 widget 값을 남은 input 슬롯에 매핑
        for name in input_order:
            if name not in inputs and widget_idx < len(widgets_values):
                inputs[name] = widgets_values[widget_idx]
                widget_idx += 1

        prompt[nid_str] = {"class_type": ntype, "inputs": inputs}
    return prompt


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
    """GET /history/{prompt_id} → 완료 시 { prompt_id: { prompt, outputs, status, meta } }, 미완료 시 {}."""
    async with httpx.AsyncClient(timeout=30.0) as client:
        r = await client.get(f"{_base()}/history/{prompt_id}")
        r.raise_for_status()
        return r.json()


async def _get_queue() -> dict[str, Any]:
    """GET /queue → { queue_running, queue_pending }. 항목은 [number, prompt_id, ...] 형태."""
    async with httpx.AsyncClient(timeout=10.0) as client:
        r = await client.get(f"{_base()}/queue")
        r.raise_for_status()
        return r.json()


def _is_prompt_in_queue(queue_data: dict[str, Any], prompt_id: str) -> bool:
    """queue_running 또는 queue_pending에 해당 prompt_id가 있는지 확인."""
    for key in ("queue_running", "queue_pending"):
        for item in queue_data.get(key) or []:
            if isinstance(item, (list, tuple)) and len(item) >= 2 and item[1] == prompt_id:
                return True
    return False


async def _get_image(filename: str, subfolder: str = "", type_: str = "output") -> bytes:
    """GET /view?filename=...&subfolder=...&type=output → image bytes."""
    params = {"filename": filename, "type": type_}
    if subfolder:
        params["subfolder"] = subfolder
    async with httpx.AsyncClient(timeout=60.0) as client:
        r = await client.get(f"{_base()}/view", params=params)
        r.raise_for_status()
        return r.content


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


# SaveVideo 등이 비디오를 images 배열로 내려줄 때 사용하는 확장자
_VIDEO_EXTENSIONS = (".mp4", ".webm", ".mov", ".avi", ".mkv", ".gif")


def _extract_first_video_from_history(history: dict[str, Any]) -> tuple[str, str, str] | None:
    """history에서 첫 비디오 출력 (SaveVideo 등). (filename, subfolder, type)."""
    if "outputs" in history:
        outputs = history["outputs"]
    else:
        for k, v in history.items():
            if isinstance(v, dict) and "outputs" in v:
                outputs = v["outputs"]
                break
        else:
            return None
    for _node_id, node_out in outputs.items():
        if not isinstance(node_out, dict):
            continue
        # 1) videos, gifs, animations, video(단일)
        videos = (
            node_out.get("videos")
            or node_out.get("gifs")
            or node_out.get("animations")
            or ([node_out["video"]] if isinstance(node_out.get("video"), dict) else None)
        )
        if videos:
            v = videos[0] if isinstance(videos[0], dict) else {}
            fn = v.get("filename")
            if fn:
                return (fn, v.get("subfolder", ""), v.get("type", "output"))
        # 2) SaveVideo 등이 비디오를 images 배열로 내려주는 경우 (예: LTX_2.3_i2v_00031_.mp4)
        images = node_out.get("images")
        if isinstance(images, list) and len(images) > 0:
            first = images[0]
            if isinstance(first, dict):
                fn = first.get("filename")
                if fn and fn.lower().endswith(_VIDEO_EXTENSIONS):
                    return (
                        fn,
                        first.get("subfolder", ""),
                        first.get("type", "output"),
                    )
                # animated: [true] 이면 비디오로 간주
                if fn and node_out.get("animated") and node_out["animated"][0:1]:
                    return (
                        fn,
                        first.get("subfolder", ""),
                        first.get("type", "output"),
                    )
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


def _inject_ltx23_workflow_inputs(workflow: dict[str, Any], image_name: str, prompt_text: str) -> dict[str, Any]:
    """워크플로에 업로드된 이미지 파일명과 프롬프트를 주입. ComfyUI 노드: LoadImage, CLIPTextEncode 등."""
    import copy
    wf = copy.deepcopy(workflow)
    for nid, node in wf.items():
        if not isinstance(node, dict):
            continue
        ct = (node.get("class_type") or "").lower()
        inputs = node.get("inputs") or {}
        if "loadimage" in ct or "load image" in ct:
            node["inputs"] = {**inputs, "image": image_name}
        if "clip" in ct and "text" in ct:
            node["inputs"] = {**inputs, "text": prompt_text}
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
) -> bytes:
    """
    LTX-2.3 이미지→비디오를 ComfyUI(LTXVideo 노드)로 실행.
    모델·파이프라인은 ComfyUI/models 및 pipelines 참조. 워크플로는 pipelines/<comfyui_ltx23_workflow>.json.
    """
    settings = get_settings()
    if not settings.comfyui_enabled:
        raise RuntimeError("ComfyUI is disabled (COMFYUI_ENABLED=false)")

    workflow_name = getattr(settings, "comfyui_ltx23_workflow", "ltx23_i2v") or "ltx23_i2v"
    workflow_path = settings.pipelines_dir / f"{workflow_name}.json"
    if not workflow_path.exists():
        workflow_path = settings.pipelines_dir / "comfyui_ltx23_workflow.json"
    if not workflow_path.exists():
        raise RuntimeError(
            f"LTX-2.3 ComfyUI workflow not found. Tried: {workflow_name}.json and comfyui_ltx23_workflow.json in {settings.pipelines_dir}. "
            "Export your LTXVideo workflow from ComfyUI and save to pipelines/ (see pipelines/README_LTX23.md)."
        )

    base_url = settings.comfyui_base_url
    try:
        raw = _load_workflow_json(workflow_path)
        # ComfyUI /prompt API expects only the prompt map (node_id -> { class_type, inputs, _meta }).
        # File may be: (1) API format with top-level "prompt" dict, or (2) top-level node objects (class_type + inputs), or (3) UI format (nodes + links).
        graph = raw.get("prompt") if isinstance(raw.get("prompt"), dict) else None
        if graph is None or len(graph) == 0:
            graph = {
                k: v
                for k, v in raw.items()
                if isinstance(v, dict) and "class_type" in v and "inputs" in v
            }
        if (not isinstance(graph, dict) or len(graph) == 0) and isinstance(raw.get("nodes"), list) and len(raw.get("nodes", [])) > 0:
            # UI 저장 형식(nodes + links): ComfyUI /object_info로 스키마를 가져와 API 형식으로 변환
            try:
                object_info = await _get_object_info()
                graph = _ui_workflow_to_prompt(raw, object_info)
            except Exception as e:
                logger.warning("UI workflow conversion failed: %s", e)
        if not isinstance(graph, dict) or len(graph) == 0:
            raise RuntimeError(
                "Workflow file must be in ComfyUI API format: include a 'prompt' key with node_id -> node_spec, "
                "or top-level node objects with 'class_type' and 'inputs', or a UI export with 'nodes' and 'links'. "
                "In ComfyUI use Save (API Format) for prompt-only JSON, or use the normal Save and ensure the file has a 'nodes' array."
            )

        uploaded = await upload_image(image_bytes)
        image_name = uploaded.get("name", "image.png")
        prompt_graph = _inject_ltx23_workflow_inputs(graph, image_name, prompt)

        out = await _post_prompt(prompt_graph)
        if "error" in out:
            raise RuntimeError(f"ComfyUI LTX-2.3 prompt error: {out['error']}")
        prompt_id = out.get("prompt_id") or str(out.get("number", ""))
        if not prompt_id:
            raise RuntimeError("ComfyUI did not return prompt_id")

        max_wait = settings.comfyui_timeout_seconds
        poll_interval = 0.5
        log_interval = 60.0
        loop = asyncio.get_event_loop()
        start = loop.time()
        last_log = start
        while (loop.time() - start) < max_wait:
            now = loop.time()
            elapsed = now - start
            history = await _get_history(prompt_id)
            # ComfyUI 응답: 완료 시 { prompt_id: { prompt, outputs, status, meta } }, 미완료 시 {}
            if isinstance(history, dict) and prompt_id in history:
                info = history[prompt_id]
                if not isinstance(info, dict):
                    info = None
            else:
                info = None
            if info is not None:
                logger.info("ComfyUI history received for %s (outputs keys: %s)", prompt_id[:8], list((info.get("outputs") or {}).keys()))
                # ComfyUI 실행 실패 시 status.status_str == "error" 또는 completed == False
                status = info.get("status") or {}
                if status.get("status_str") == "error" or status.get("completed") is False:
                    messages = status.get("messages") or []
                    msg = "; ".join(str(m) for m in messages) if messages else "ComfyUI workflow failed (no message)."
                    raise RuntimeError(f"ComfyUI workflow error: {msg}")
                first_video = _extract_first_video_from_history(info)
                if first_video:
                    filename, subfolder, type_ = first_video
                    return await _get_video_bytes(filename, subfolder, type_)
                # 완료됐는데 비디오 없음 → 출력 노드 형식이 다르거나 워크플로 문제
                raise RuntimeError(
                    "ComfyUI workflow finished but no video output found. "
                    "Ensure the workflow has a SaveVideo (or similar) node and its output is in history outputs."
                )
            if (now - last_log) >= log_interval:
                try:
                    queue_data = await _get_queue()
                    in_queue = _is_prompt_in_queue(queue_data, prompt_id)
                    logger.info(
                        "ComfyUI workflow %s (%.0fs / %.0fs) — %s",
                        prompt_id[:8],
                        elapsed,
                        max_wait,
                        "in queue (running/pending)" if in_queue else "not in queue (waiting for history or may have failed)",
                    )
                except Exception as e:
                    logger.warning("ComfyUI queue check failed: %s", e)
                last_log = now
            await asyncio.sleep(poll_interval)

        raise TimeoutError(
            f"ComfyUI LTX-2.3 workflow {prompt_id} did not finish within {max_wait}s. "
            f"Increase COMFYUI_TIMEOUT_SECONDS (current: {max_wait}) if your workflow needs more time."
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
        msg = f"ComfyUI returned {e.response.status_code if e.response else 'error'} for /prompt."
        if body:
            msg += f" Response: {body}"
        raise RuntimeError(msg) from e
