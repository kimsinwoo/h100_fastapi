"""
Universal Animal Image Generation Engine — Pose Lock & Conditional Augmentation.

Phase 2: Pose lock base (pose, camera, gravity, structure).
Phase 3: Conditional augmentation (view_angle, body_pose).
Phase 4: Clothing handling.
Phase 6: Style injection (after pose rules; never overrides).
Phase 7: Validation comparison (detect drift for retry).

Fully modular; no hardcoded species logic; extensible for new poses.
"""

from __future__ import annotations

from typing import Any

# ---------------------------------------------------------------------------
# Phase 2 — Pose Lock Base
# ---------------------------------------------------------------------------

POSE_LOCK_BASE = (
    "POSE LOCK: Preserve exact pose. Do not rotate body. Do not adjust posture. "
    "Do not symmetrize face. Do not center head. "
    "CAMERA LOCK: Preserve camera angle. Preserve number of visible eyes. "
    "Preserve limb visibility. Preserve orientation. "
    "GRAVITY LOCK: Preserve gravity direction. Do not normalize to upright. "
    "STRUCTURE RULES: Do not add limbs. Do not remove visible limbs. "
    "No perspective correction. No frontal bias. No symmetry correction. "
    "If pose changes, regenerate."
)

# PART 4: Full negative enforcement (structure + lighting + style)
POSE_LOCK_NEGATIVE = (
    "frontal correction, symmetry fix, pose adjustment, extra limbs, missing limbs, "
    "perspective normalization, rotate body, adjust posture, symmetrize face, "
    "normalize to upright, add limbs, remove limbs, center head, "
    "dark dramatic lighting, strong contrast, moody shadows, oversaturated colors, "
    "clothing blended into fur, garment removed, outfit added when none, overexposure, harsh shadows"
)

# ---------------------------------------------------------------------------
# Phase 3 — Conditional Augmentation (view_angle, body_pose)
# ---------------------------------------------------------------------------

def _view_angle_block(view_angle: str) -> str:
    """Strict rules by view. side-left / side-right / rear."""
    va = (view_angle or "").strip().lower()
    if va in ("side-left", "side_left", "side-profile-left"):
        return (
            "Strict 90-degree side profile. Only one eye visible. No frontal exposure."
        )
    if va in ("side-right", "side_profile_right", "side-profile-right"):
        return (
            "Strict 90-degree side profile. Only one eye visible. No frontal exposure."
        )
    if va == "rear":
        return (
            "Rear view composition. Face must not be visible. No head rotation."
        )
    return ""


def _body_pose_block(body_pose: str) -> str:
    """Lying / jumping augmentation."""
    bp = (body_pose or "").strip().lower()
    if bp == "lying":
        return (
            "Body horizontal. Spine parallel to ground. No standing correction."
        )
    if bp == "jumping":
        return "Mid-air pose. Do not force ground contact."
    return ""


def build_conditional_augmentation(analysis: dict[str, Any] | Any) -> str:
    """
    Phase 3. Returns augmentation block from view_angle and body_pose.
    analysis can be UniversalAnalysisResponse (model_dump()) or dict.
    """
    if hasattr(analysis, "model_dump"):
        analysis = analysis.model_dump()
    if not isinstance(analysis, dict):
        return ""
    parts = []
    view_block = _view_angle_block(analysis.get("view_angle") or "")
    if view_block:
        parts.append(view_block)
    pose_block = _body_pose_block(analysis.get("body_pose") or "")
    if pose_block:
        parts.append(pose_block)
    return " ".join(parts)


# ---------------------------------------------------------------------------
# Phase 4 — Clothing Handling (high-precision; structure preservation)
# ---------------------------------------------------------------------------

# PART 2: Mandatory blocks — clothing is structural element or no clothing allowed
CLOTHING_PRESERVATION_WHEN_PRESENT = (
    "CLOTHING IS A STRUCTURAL ELEMENT. "
    "Clothing must remain separate from fur. "
    "Do not blend clothing into body texture. "
    "Do not remove garment. "
    "Maintain fabric folds and boundaries."
)

CLOTHING_PRESERVATION_WHEN_ABSENT = (
    "No clothing allowed. "
    "Do not add shirts, sweaters, costumes, or outfits. "
    "Body must remain fully natural."
)


def build_clothing_rules(
    is_wearing_clothes: bool,
    clothing_type: str = "",
    clothing_color: str = "",
    clothing_pattern: str = "",
    clothing_coverage: str = "",
) -> str:
    """
    PART 2. Clothing structure preservation. If wearing clothes: mandatory structural block.
    If not: no clothing allowed. Collar/harness alone do NOT count as clothing.
    """
    if is_wearing_clothes:
        parts = [CLOTHING_PRESERVATION_WHEN_PRESENT]
        if clothing_type and clothing_type.lower() != "none":
            parts.append(f"Clothing type: {clothing_type}.")
        if clothing_color:
            parts.append(f"Clothing color: {clothing_color}.")
        if clothing_pattern:
            parts.append(f"Clothing pattern: {clothing_pattern}.")
        if clothing_coverage and clothing_coverage.lower() not in ("none", ""):
            parts.append(f"Coverage: {clothing_coverage}.")
        return " ".join(parts)
    return CLOTHING_PRESERVATION_WHEN_ABSENT


def _get_clothing_block(analysis: dict[str, Any] | Any) -> str:
    if hasattr(analysis, "model_dump"):
        analysis = analysis.model_dump()
    if not isinstance(analysis, dict):
        return build_clothing_rules(False)
    return build_clothing_rules(
        bool(analysis.get("is_wearing_clothes", False)),
        str(analysis.get("clothing_type") or ""),
        str(analysis.get("clothing_color") or ""),
        str(analysis.get("clothing_pattern") or ""),
        str(analysis.get("clothing_coverage") or ""),
    )


# ---------------------------------------------------------------------------
# Phase 2 + 3 + 4 + 6 — Full pose-lock prompt (style injected after pose rules)
# ---------------------------------------------------------------------------

def build_pose_lock_prompt(
    analysis: dict[str, Any] | Any,
    style_block: str | None = None,
    subject_prefix: str | None = None,
) -> str:
    """
    Build full pose-lock prompt: base + conditional augmentation + clothing + optional subject + style.
    Style block is appended last and must never override pose rules.
    """
    if hasattr(analysis, "model_dump"):
        analysis = analysis.model_dump()
    if not isinstance(analysis, dict):
        analysis = {}

    parts = [POSE_LOCK_BASE]
    aug = build_conditional_augmentation(analysis)
    if aug:
        parts.append(aug)
    parts.append(_get_clothing_block(analysis))
    if subject_prefix and subject_prefix.strip():
        parts.append(subject_prefix.strip())
    if style_block and style_block.strip():
        parts.append(style_block.strip())
    return " ".join(parts)


def build_pose_lock_negative() -> str:
    """Phase 5. Negative prompt for pose/structure preservation."""
    return POSE_LOCK_NEGATIVE


# ---------------------------------------------------------------------------
# Phase 5 — Generation params (caller uses these)
# ---------------------------------------------------------------------------

def get_pose_lock_strength(pose_rarity: str = "normal") -> float:
    """Strength 0.65–0.75 depending on pose rarity. normal=0.70."""
    r = (pose_rarity or "normal").strip().lower()
    if r in ("rare", "unusual"):
        return 0.68
    if r in ("common", "default"):
        return 0.72
    return 0.70


def get_pose_lock_guidance_min() -> float:
    """Guidance scale >= 8."""
    return 8.0


# ---------------------------------------------------------------------------
# Phase 7 — Validation: compare analyses (drift => regenerate)
# ---------------------------------------------------------------------------

def analysis_drift_requires_retry(
    initial: dict[str, Any] | Any,
    re_analyzed: dict[str, Any] | Any,
) -> bool:
    """
    Returns True if re_analyzed differs on critical fields so we should regenerate.
    - visible_eyes changed
    - body_pose changed
    - gravity_axis normalized (e.g. was rotated, now normal)
    - is_wearing_clothes changed (clothing disappeared or appeared)
    - leg_visibility_count changed
    - view_angle changed (e.g. side -> front)
    """
    def _to_dict(a: Any) -> dict[str, Any]:
        if hasattr(a, "model_dump"):
            return a.model_dump()
        return dict(a) if isinstance(a, dict) else {}

    i = _to_dict(initial)
    r = _to_dict(re_analyzed)

    if i.get("visible_eyes") != r.get("visible_eyes"):
        return True
    if (i.get("body_pose") or "").strip() != (r.get("body_pose") or "").strip():
        return True
    if (i.get("gravity_axis") or "").strip() != (r.get("gravity_axis") or "").strip():
        return True
    if i.get("is_wearing_clothes") != r.get("is_wearing_clothes"):
        return True
    if i.get("leg_visibility_count") != r.get("leg_visibility_count"):
        return True
    if (i.get("view_angle") or "").strip() != (r.get("view_angle") or "").strip():
        return True
    return False


def validation_requires_retry(
    initial: dict[str, Any] | Any,
    re_analyzed: dict[str, Any] | Any,
    *,
    required_cloud_theme: bool = False,
) -> bool:
    """
    PART 5. Validation loop: retry if pose/structure/clothing/cloud integrity failed.
    - Pose/structure drift (delegate to analysis_drift_requires_retry).
    - Clothing disappeared, clothing added when none, clothing blended into fur (confidence drop).
    - required_cloud_theme: when True, future scene classifier should check (GPT Cloud Replica):
      1. Background fully cloud-dominant? 2. Lighting high-key and low contrast?
      3. Pet anatomy unchanged? 4. No dark shadows? 5. Scene airy and weightless?
      If any fail → regenerate.
    - Background override: when segment→composite was used, original background is removed programmatically.
      If replacement was skipped (e.g. rembg unavailable) and scene classifier detects original background visible → regenerate.
    """
    if analysis_drift_requires_retry(initial, re_analyzed):
        return True
    def _to_dict(a: Any) -> dict[str, Any]:
        if hasattr(a, "model_dump"):
            return a.model_dump()
        return dict(a) if isinstance(a, dict) else {}
    i = _to_dict(initial)
    r = _to_dict(re_analyzed)
    # Clothing blended: initial had clothes with high confidence, re-analysis shows low confidence
    if i.get("is_wearing_clothes") and r.get("is_wearing_clothes") is False:
        return True  # clothing disappeared
    if not i.get("is_wearing_clothes") and r.get("is_wearing_clothes"):
        return True  # clothing added when none
    ci = float(i.get("confidence_score") or 0)
    cr = float(r.get("confidence_score") or 0)
    if i.get("is_wearing_clothes") and ci >= 0.6 and cr < 0.4:
        return True  # likely blended into fur
    # Cloud theme presence check: not implemented (no scene classifier); reserved for future
    if required_cloud_theme:
        pass
    return False
