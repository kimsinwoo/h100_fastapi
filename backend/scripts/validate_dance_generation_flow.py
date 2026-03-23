#!/usr/bin/env python3
"""
Validate dance generation integration (imports, workflow JSON load, schema).
Run from backend:  PYTHONPATH=. python scripts/validate_dance_generation_flow.py
"""

from __future__ import annotations

import sys
from pathlib import Path

# backend/ as cwd
_BACKEND = Path(__file__).resolve().parent.parent
if str(_BACKEND) not in sys.path:
    sys.path.insert(0, str(_BACKEND))


def main() -> int:
    from app.core.config import get_settings
    from app.services.comfyui_service import inject_dance_pose_workflow_inputs, load_dance_workflow_template
    from app.services.dance_generation_service import DanceGenerationService

    svc = DanceGenerationService()
    assert svc is not None

    wf_path = get_settings().pipelines_dir / "dance" / "dog_pose_generation.json"
    if not wf_path.is_file():
        print("SKIP: missing", wf_path)
        return 0

    wf = load_dance_workflow_template("dog_pose_generation.json")
    injected = inject_dance_pose_workflow_inputs(
        wf,
        character_image_name="c.png",
        pose_image_name="p.png",
        seed=123,
        positive_prompt="test pos",
        negative_prompt="test neg",
    )
    assert isinstance(injected, dict) and len(injected) >= 1
    print("validate_dance_generation_flow: OK (workflow nodes:", len(injected), ")")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
