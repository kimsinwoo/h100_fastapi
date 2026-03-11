"""
ComfyUI 연동: 로컬 ComfyUI 서버(기본 8188)에 워크플로우를 보내고 결과 이미지를 반환.
backend 루트 기준 app, static, scripts, motions, pipelines 사용.
"""

from __future__ import annotations

import asyncio
import logging
import uuid
from pathlib import Path
from typing import Any

import httpx

from app.core.config import get_settings
from app.utils.file_handler import ensure_generated_dir

logger = logging.getLogger(__name__)


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
    """GET /history/{prompt_id} → execution result with outputs."""
    async with httpx.AsyncClient(timeout=30.0) as client:
        r = await client.get(f"{_base()}/history/{prompt_id}")
        r.raise_for_status()
        data = r.json()
        return data.get(prompt_id, data)


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
        videos = node_out.get("videos") or node_out.get("gifs") or node_out.get("animations")
        if not videos:
            continue
        v = videos[0] if isinstance(videos[0], dict) else {}
        fn = v.get("filename")
        if fn:
            return (fn, v.get("subfolder", ""), v.get("type", "output"))
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
        raise RuntimeError(
            f"LTX-2.3 ComfyUI workflow not found: {workflow_path}. "
            "Export your LTXVideo workflow from ComfyUI and save as pipelines/ltx23_i2v.json (see pipelines/README_LTX23.md)."
        )

    base_url = settings.comfyui_base_url
    try:
        import json
        raw = json.loads(workflow_path.read_text(encoding="utf-8"))
        graph = raw.get("prompt") if isinstance(raw.get("prompt"), dict) else raw

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
        start = asyncio.get_event_loop().time()
        while (asyncio.get_event_loop().time() - start) < max_wait:
            history = await _get_history(prompt_id)
            if prompt_id in history:
                info = history[prompt_id]
                if isinstance(info, dict):
                    first_video = _extract_first_video_from_history(info)
                    if first_video:
                        filename, subfolder, type_ = first_video
                        return await _get_video_bytes(filename, subfolder, type_)
            await asyncio.sleep(poll_interval)

        raise TimeoutError(f"ComfyUI LTX-2.3 workflow {prompt_id} did not finish within {max_wait}s")
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
