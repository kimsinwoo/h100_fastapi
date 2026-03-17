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
        # ComfyUI 노드별 키: videos, gifs, animations, animated(LTX 등), images(단일 비디오인 경우도 있음)
        raw = (
            node_out.get("videos")
            or node_out.get("gifs")
            or node_out.get("animations")
            or node_out.get("animated")
            or (node_out.get("video") if isinstance(node_out.get("video"), list) else None)
            or node_out.get("images")
        )
        # 리스트가 아니면 단일 항목(딕셔너리/문자열)으로 취급
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
                sub = inner.get("subfolder", "")
                typ = inner.get("type", "output")
                return (fn, sub, typ)
            # 구조 확인용: 실제 키와 값 일부 로그 (노드 273 등 animated 형식 대응)
            logger.warning(
                "ComfyUI output 항목에 filename/name/path 없음. 키=%s, 값예시=%s",
                list(first.keys())[:12],
                {k: (v[:80] if isinstance(v, str) else v) for k, v in list(first.items())[:5]},
            )
        elif isinstance(first, str):
            return (first, "", "output")
        elif isinstance(first, (list, tuple)) and len(first) >= 1:
            # [filename, subfolder, type] 형태
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
        # Full workflow JSON has "prompt", "last_node_id", "version", "nodes", "links" etc.; send only "prompt".
        graph = raw.get("prompt") if isinstance(raw.get("prompt"), dict) else raw
        if not isinstance(graph, dict):
            raise RuntimeError(
                "Workflow file must be in ComfyUI API format: include a 'prompt' key with node_id -> node_spec. "
                "In ComfyUI use Save (API Format) or export the prompt-only JSON."
            )
        # When we used full raw (no "prompt" key), ensure it is prompt-only (every value is a node dict)
        if graph is raw:
            for k, v in graph.items():
                if not isinstance(v, dict) or "class_type" not in v or "inputs" not in v:
                    raise RuntimeError(
                        "Workflow file must be in ComfyUI API format (prompt-only). "
                        "Save your workflow in ComfyUI as 'Save (API Format)' and use that JSON."
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

        max_wait = getattr(settings, "comfyui_video_timeout_seconds", None) or settings.comfyui_timeout_seconds
        poll_interval = 2.0  # 0.5→2초: 로그·부하 감소, 영상 워크플로는 완료까지 시간 걸림
        start = asyncio.get_event_loop().time()
        last_log_at = 0.0
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
                # 완료됐는데 비디오 못 찾은 경우: 노드별 출력 키 로그 (노드 구조가 다르면 여기서 확인)
                outputs = info.get("outputs") if isinstance(info.get("outputs"), dict) else {}
                node_keys = list(outputs.keys())
                logger.warning(
                    "ComfyUI LTX-2.3 완료로 보이지만 비디오 출력을 찾지 못함. output 노드 수=%s, 노드별 키 예시=%s",
                    len(node_keys),
                    {nid: list(v.keys()) if isinstance(v, dict) else type(v).__name__ for nid, v in list(outputs.items())[:3]},
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
        msg = f"ComfyUI returned {e.response.status_code if e.response else 'error'} for /prompt."
        if body:
            msg += f" Response: {body}"
        raise RuntimeError(msg) from e
