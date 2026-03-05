# Universal Animal Image Generation Engine — Architecture

## Overview

Modular, pose-stable img2img pipeline: analyze → pose-lock prompt → generate → (optional) validate & retry.

## Phase 1 — Image Analysis Module

- **Schema**: `UniversalAnalysisResponse` (species, view_angle, body_pose, gravity_axis, head_direction_degrees, spine_alignment, visible_eyes, leg_visibility_count, is_full_body_visible, is_wearing_clothes, clothing_*).
- **API**: `POST /api/image/analyze-universal` → JSON only. Stub: form overrides or defaults; plug in vision later.
- **Rules**: No auto-correct pose, no assumed symmetry, collar ≠ clothing.

## Phase 2 — Pose Lock Engine

- **Module**: `app.utils.pose_lock_engine`
  - `build_pose_lock_base()` → POSE LOCK + CAMERA LOCK + GRAVITY LOCK + STRUCTURE RULES.
  - `build_conditional_augmentation(analysis)` → view_angle (side/rear) and body_pose (lying/jumping) blocks.
  - `build_clothing_rules(analysis)` → mandatory clothing vs no clothing.
  - `build_pose_lock_prompt(analysis, style_block)` → base + augmentation + clothing + style (style never overrides pose).
  - `build_pose_lock_negative()` → frontal correction, symmetry, pose adjustment, limbs, perspective.
- **Typing**: Analysis as `UniversalAnalysisResponse` or dict; no hardcoded species.

## Phase 3 — Conditional Augmentation

- side-left / side-right → strict 90° side profile, one eye, no frontal exposure.
- rear → rear view, face not visible, no head rotation.
- lying → body horizontal, spine parallel to ground.
- jumping → mid-air, no force ground contact.

## Phase 4 — Clothing Handling

- If `is_wearing_clothes`: clothing mandatory, separate from fur, no convert to body texture.
- If not: no clothing allowed, no outfits/costumes.

## Phase 5 — Generation Settings

- img2img; strength 0.65–0.75 (pose rarity can tune); guidance_scale >= 8.
- Disable face correction; enable pose preservation.
- Negative: frontal correction, symmetry fix, pose adjustment, extra/missing limbs, perspective normalization.

## Phase 6 — Style Injection

- Style block injected **after** pose-lock rules in prompt.
- Style never overrides pose/camera/gravity/structure rules.

## Phase 7 — Validation Loop

- After generation, re-analyze output (same JSON structure).
- If visible_eyes, pose, gravity, clothing, or limb count changed → regenerate (max retries configurable).
- `compare_analysis(initial, re_analyzed)` → True if regeneration required.

## Code Structure

```
app/
  schemas/
    image_schema.py          # + UniversalAnalysisResponse
  utils/
    prompt_builder.py        # existing style prompts
    pose_lock_engine.py       # NEW: pose lock + augmentation + clothing + validation
  services/
    image_service.py         # + run_universal_animal_generate(analysis, style_key, ...) + validation retry
  api/
    routes.py                # POST /api/image/analyze-universal, POST /api/generate with optional analysis
```

## Data Flow

1. Client uploads image → `POST /api/image/analyze-universal` → `UniversalAnalysisResponse`.
2. Client calls `POST /api/generate` with same image + `analysis` (JSON) + `style` + `use_pose_lock=true`.
3. Server builds prompt via `build_pose_lock_prompt(analysis, style_block)`; strength 0.65–0.75; guidance >= 8.
4. If `validate_and_retry=true`: after first image, re-analyze (stub), compare; if drift, regenerate once.
