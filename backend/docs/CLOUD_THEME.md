# Cloud Theme Rendering Module

High-key, ultra-soft cloud-themed rendering for z-image-turbo. Preserves subject structure; no overexposure, over-stylization, or contrast distortion.

## Module: `app.utils.cloud_theme`

### 1. Inject into any base prompt (safe merge)

```python
from app.utils.cloud_theme import inject_cloud_theme_into_prompt

# base_prompt may be pose-lock or any existing prompt; cloud is appended and never overrides
combined = inject_cloud_theme_into_prompt(base_prompt="...", intensity="medium")
```

- **intensity**: `"low"` (subtle), `"medium"` (balanced), `"high"` (strong dreamy effect, no structural distortion).
- Cloud style is always appended after the base; pose-lock instructions in the base remain first and are never replaced.

### 2. Standalone style block (with structure protection)

```python
from app.utils.cloud_theme import get_cloud_theme_block, get_cloud_theme_negative

prompt_block = get_cloud_theme_block("medium")  # structure protection + style
negative = get_cloud_theme_negative()
```

### 3. Style-only block (for custom pipelines)

```python
from app.utils.cloud_theme import get_cloud_theme_style_block

style_only = get_cloud_theme_style_block("high")
```

## Structure protection (always included in block)

- Preserve exact pose, visible limbs, visible eyes, orientation.
- No pose correction, symmetry correction, perspective normalization, or anatomy alteration.

## Generation settings (z-image-turbo)

- **Style key**: `cloud_theme` / `cloud theme`
- **Strength**: 0.48–0.54 (structure stability over artistic exaggeration)
- **Guidance**: 7.0 (balanced; not extreme)
- **Steps**: 36

## Pose-lock integration

When using the Universal Animal engine with `style_key="cloud_theme"`, pass optional `cloud_intensity` (`low` / `medium` / `high`) to `run_universal_animal_generate` or set it in the style block via `get_style_block(style_key, cloud_intensity=...)`.
