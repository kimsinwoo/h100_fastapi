"""
Stylized 2D character redesign for ANY pet species.
Subject: small pet animal (dog, cat, rabbit, bird, etc.) → simplified 2D animated character.
Never preserve biological realism. No fur/feather/skin realism, no 3D lighting.
"""

from __future__ import annotations

import random
from typing import Any

# ========== BASE PROMPT (species-agnostic) ==========
# Stylized 2D animated character redesign; subject = "small pet animal character"
BASE_PROMPT = (
    "stylized 2D animated character redesign of a small pet animal character, "
    "simplified shape language, clear silhouette, flat color areas, limited shading layers, "
    "no biological realism, no realistic muscle definition, no real texture replication, "
    "style-locked rendering, not a photograph, animated character form only"
)

# ========== BASE NEGATIVE ==========
BASE_NEGATIVE = (
    "photorealistic, real fur, real feathers, real skin texture, "
    "cinematic lighting, 3D rendering, ray tracing, depth of field, subsurface scattering, "
    "hyper detailed, realistic anatomy, real muscle structure, high frequency texture, HDR"
)

# ========== POSE PRESERVATION (참조 이미지 자세 유지: 누움/앉음/서 있음 등) ==========
# img2img 시 원본 사진의 자세·구도를 바꾸지 않도록 명시
POSE_PRESERVATION = (
    "exact same pose and body position as reference image, "
    "preserve composition and posture, same lying or sitting or standing as input, "
    "do not change pose, keep body orientation and limb arrangement identical"
)
POSE_NEGATIVE = (
    "different pose, changed posture, wrong body position, altered composition, "
    "standing when reference is lying, lying when reference is standing, sitting when reference is lying, "
    "reposed, pose change, different angle"
)

# ========== SPECIES SUBJECT (주어 명시: 고양이/강아지 구분 확실히) ==========
# 프롬프트 맨 앞에 올려 모델이 종을 명확히 인식하도록 함.
SPECIES_SUBJECT: dict[str, str] = {
    "dog": "dog character",
    "cat": "cat character",
    "rabbit": "rabbit character",
    "hamster": "hamster character",
    "ferret": "ferret character",
    "bird": "bird character",
    "turtle": "turtle character",
    "reptile": "reptile character",
    "pet": "small pet animal character",
}

# ========== SPECIES ADAPTATION LAYER ==========
# Silhouette, ear, eye, limb, tail, texture simplification per species.
SPECIES_MODIFIERS: dict[str, str] = {
    "dog": (
        "rounded muzzle silhouette, simplified ear blocks, tail as single curved shape, "
        "no fur strands, simplified limb blocks, no wet nose detail"
    ),
    "cat": (
        "pointed triangular upright ears, both ears with visible pink inner ear symmetrical, "
        "sharper eye shape slit pupils, small nose no long snout, tail as smooth tapered curve, "
        "whiskers as simple lines only no realism, simplified limb, no fur strands, feline not canine"
    ),
    "rabbit": (
        "elongated ear shapes, minimal facial detail, round body mass, "
        "simplified feet, no fur texture, clear silhouette"
    ),
    "hamster": (
        "round compact silhouette, small simplified ears, minimal limb detail, "
        "no fur strands, simple cheek shape, no realistic fur texture"
    ),
    "ferret": (
        "elongated body as simplified shape, rounded ears, tapered tail as single curve, "
        "no fur detail, simplified limbs, clear silhouette"
    ),
    "bird": (
        "beak as flat geometric wedge, wings as layered flat shapes, no feather detail, "
        "feet simplified to minimal shapes, round or simplified head, no plumage texture"
    ),
    "turtle": (
        "shell as simple dome or rounded shape, head and limbs as simplified blocks, "
        "no scale texture, no realistic skin, clear silhouette"
    ),
    "reptile": (
        "simplified body silhouette, head as geometric shape, no scale texture, "
        "limbs as simple shapes, no realistic skin detail"
    ),
    "pet": (
        "simplified animal silhouette, clear shape language, no fur or feather strands, "
        "minimal limb and tail detail, flat color regions, readable at small size"
    ),
}

# ========== SPECIES NEGATIVE (반대 종 회피: 고양이일 때 강아지 형태 차단, 강아지일 때 고양이 형태 차단) ==========
SPECIES_NEGATIVE_AVOID: dict[str, str] = {
    "dog": "cat, feline, cat-like, triangular cat ears, slit pupils, cat silhouette",
    "cat": (
        "dog, canine, dog-like, rounded dog muzzle, floppy dog ears, dog silhouette, "
        "rounded ears, long snout, only one ear visible, asymmetrical ears, one ear empty"
    ),
    "rabbit": "dog, cat, canine, feline, long dog muzzle",
    "hamster": "dog, cat, long ears, long muzzle",
    "ferret": "dog, cat, bird, beak",
    "bird": "dog, cat, mammal, fur, four legs",
    "turtle": "dog, cat, fur, ears",
    "reptile": "dog, cat, fur, fluffy",
    "pet": "",
}

# ========== STYLE LAYER (construction rules) ==========
# Outline thickness, shading layer count, color palette, eye rendering, highlight, shadow geometry.
STYLE_PROMPTS: dict[str, str] = {
    "dragonball": (
        "thick 3-4px black outline, 2-tone cel shading, single hard shadow block, "
        "flat color fill no gradient, bold simplified anatomy, large round eyes with single highlight, "
        "limited color palette, no texture, manga panel look"
    ),
    "slamdunk": (
        "medium 2-3px outline, 2-layer cel shading, angular shadow blocks, "
        "flat color regions, sharp determined eyes with 2-tone, "
        "dynamic pose construction, sports manga linework, no gradient, no soft shadow"
    ),
    "sailor_moon": (
        "clean 2px outline, 2-tone cel shading, sparkle and star motifs, "
        "flat pastel color fill, large eyes with multiple highlights, "
        "ribbon and simple costume shapes, no gradient, 90s anime cel"
    ),
    "pokemon": (
        "soft 1-2px outline, 2-layer cel shading, rounded shadow shapes, "
        "flat saturated color, cute proportion, large simple eyes with one highlight, "
        "rounded silhouette, creature design flat color, no gradient"
    ),
    "dooly": (
        "thick 2-3px outline, 1-tone flat fill, minimal or no shadow, "
        "flat color only, simple rounded anatomy, dot eyes or very simple eyes, "
        "comic strip style, limited palette, no gradient, no shading layers"
    ),
    "mazinger": (
        "thick mechanical outline 3-4px, 2-tone cel shading, hard angular shadows, "
        "flat color with edge highlight only, simplified mecha silhouette, "
        "retro super robot style, no gradient, blocky shadow geometry"
    ),
    "shinchan": (
        "rough 2px outline, 1-2 tone flat shading, simple angular shadow, "
        "flat color fill, exaggerated cartoon proportion, minimal eye detail, "
        "gag manga style, limited palette, no gradient, simple shape-based"
    ),
    "pixel_art": (
        "masterpiece best quality, 16-bit era character sprite, "
        "single pixel line width, 0 gradient, pixel cluster shading only, "
        "hard pixel edges only, no anti-aliasing, max 16 colors, no soft shading, "
        "sprite composition, clear readable silhouette at 1x zoom, square pixels, bold outline"
    ),
    "pixel art": (
        "masterpiece best quality, 16-bit era character sprite, "
        "single pixel line width, 0 gradient, pixel cluster shading only, "
        "hard pixel edges only, no anti-aliasing, max 16 colors, no soft shading, "
        "sprite composition, clear readable silhouette at 1x zoom, square pixels, bold outline"
    ),
    # 동물의숲 단일 스타일: 완전 재디자인 (해부/비율 보존 안 함, 배경 무시, strength 0.78, pose OFF)
    "animal_crossing": (
        "Completely redesign this character into an Animal Crossing villager. Do NOT preserve original anatomy. Do NOT preserve original proportions. "
        "Ignore original background entirely. Rebuild the character with: large head (50% of body height), short cylindrical torso, "
        "short rounded limbs, simplified mitten-like hands, compact feet, oversized oval eyes, minimal facial detail, "
        "smooth stylized 3D Nintendo render. Place the character in a fully generated Animal Crossing village: "
        "symmetrical dirt path, flower beds, fruit trees, cozy house, bright daytime lighting. "
        "Preserve only: fur color pattern, eye color, basic species identity. Everything else redesigned in Animal Crossing 3D style. "
        "Species-specific features (dog: rounded ears, short snout; cat: pointed triangular ears, slit pupils; "
        "rabbit: long ears, fluffy tail; hamster: plump cheeks; bird: colorful feathers). "
        "Background: {{AC_BACKGROUND}}. Character and background fully consistent Animal Crossing 3D."
    ),
    "animal crossing": (
        "Completely redesign this character into an Animal Crossing villager. Do NOT preserve original anatomy. Do NOT preserve original proportions. "
        "Ignore original background entirely. Rebuild the character with: large head (50% of body height), short cylindrical torso, "
        "short rounded limbs, simplified mitten-like hands, compact feet, oversized oval eyes, minimal facial detail, "
        "smooth stylized 3D Nintendo render. Place the character in a fully generated Animal Crossing village: "
        "symmetrical dirt path, flower beds, fruit trees, cozy house, bright daytime lighting. "
        "Preserve only: fur color pattern, eye color, basic species identity. Everything else redesigned in Animal Crossing 3D style. "
        "Species-specific features (dog: rounded ears, short snout; cat: pointed triangular ears, slit pupils; "
        "rabbit: long ears, fluffy tail; hamster: plump cheeks; bird: colorful feathers). "
        "Background: {{AC_BACKGROUND}}. Character and background fully consistent Animal Crossing 3D."
    ),
    # Hybrid: preserve 3D quality/lighting from ref; force AC village + villager proportions (fixed village, strength 0.48, guidance 8)
    "animal_crossing_hybrid": (
        "Completely ignore and remove original image background. Do not reuse any elements from original environment. "
        "Fully modeled Animal Crossing village scene: centered dirt path with stone borders, symmetrical flower beds on both sides, "
        "colorful tulips and seasonal flowers, fruit trees in background (apples, peaches), cozy house with orange roof, "
        "soft blue sky, wooden fence, bright Nintendo-style daytime lighting. "
        "Character standing naturally on the ground aligned to the path perspective. Shadows match ground direction. "
        "Camera angle slightly frontal, eye-level, centered composition. "
        "Body proportion: Animal Crossing villager anatomy. Head size 45-50% of total body height. Body small and cylindrical. "
        "Arms short and rounded. Legs short and stubby. Hands simplified mitten-like. Feet rounded and compact. "
        "No realistic cat anatomy, no long torso, no realistic limb joints. "
        "Facial structure: large oval eyes (40% wider than realistic), small triangle nose, simple mouth, smooth cheeks, "
        "no realistic fur detail, minimal whisker thickness, clean symmetrical ears. "
        "Cat villager: upright triangular ears, pink inner ears, slit pupils, compact rounded snout, tail short and simple curve. "
        "No realistic feline bone structure. "
        "Soft 3D shading, subsurface scattering feel, smooth high-resolution textures, global illumination lighting, "
        "soft ambient occlusion, polished Nintendo-like 3D rendering, rounded geometry, no harsh contrast, no painterly strokes. "
        "Modern 3D game asset, Nintendo-style polished render, high quality stylized PBR but simplified."
    ),
    "animal crossing hybrid": (
        "Completely ignore and remove original image background. Do not reuse any elements from original environment. "
        "Fully modeled Animal Crossing village scene: centered dirt path with stone borders, symmetrical flower beds on both sides, "
        "colorful tulips and seasonal flowers, fruit trees in background (apples, peaches), cozy house with orange roof, "
        "soft blue sky, wooden fence, bright Nintendo-style daytime lighting. "
        "Character standing naturally on the ground aligned to the path perspective. Shadows match ground direction. "
        "Camera angle slightly frontal, eye-level, centered composition. "
        "Body proportion: Animal Crossing villager anatomy. Head size 45-50% of total body height. Body small and cylindrical. "
        "Arms short and rounded. Legs short and stubby. Hands simplified mitten-like. Feet rounded and compact. "
        "No realistic cat anatomy, no long torso, no realistic limb joints. "
        "Facial structure: large oval eyes (40% wider than realistic), small triangle nose, simple mouth, smooth cheeks, "
        "no realistic fur detail, minimal whisker thickness, clean symmetrical ears. "
        "Cat villager: upright triangular ears, pink inner ears, slit pupils, compact rounded snout, tail short and simple curve. "
        "No realistic feline bone structure. "
        "Soft 3D shading, subsurface scattering feel, smooth high-resolution textures, global illumination lighting, "
        "soft ambient occlusion, polished Nintendo-like 3D rendering, rounded geometry, no harsh contrast, no painterly strokes. "
        "Modern 3D game asset, Nintendo-style polished render, high quality stylized PBR but simplified."
    ),
}

# 동물의숲 3D: 배경 랜덤 선택 (원본 이미지와 무관, 게임 배경과 캐릭터 3D 통일)
AC_BACKGROUNDS: list[str] = [
    "village path with flowers and trees",
    "town plaza with cobblestone paths and wooden bridge",
    "flower field with gentle hills",
    "river bank with trees and flowers",
    "forested area with bushes and rocks",
]
DEFAULT_AC_BACKGROUND = "village path with flowers and trees"

# 4-stage AC Villager pipeline: Stage 1 allowed species (lowercase only)
AC_PIPELINE_SPECIES = ("cat", "dog", "rabbit", "hamster", "bird", "other")

# Stage 2: strict base villager anatomy (mandatory)
AC_PIPELINE_BASE_ANATOMY = (
    "Animal Crossing villager, strict proportions: head exactly 50% of total height, body short and cylindrical, "
    "legs very short and stubby, arms small and rounded, mitten-like hands, rounded compact feet, "
    "oversized vertical oval eyes, flat simplified face, no realistic anatomy, no realistic fur physics, no muscles, no real animal proportions. "
)

# Stage 2: species adaptation (cat, dog, rabbit, hamster, bird; other -> generic)
AC_PIPELINE_SPECIES_ADAPT: dict[str, str] = {
    "cat": "Cat villager: upright triangular ears, small rounded snout, no detailed whiskers.",
    "dog": "Dog villager: short rounded snout, simplified geometric ears, short stylized tail.",
    "rabbit": "Rabbit villager: thick upright rounded ears, small oval nose, short cotton tail simplified.",
    "hamster": "Hamster villager: slightly larger head ratio (55%), very small limbs, rounded cheeks, tiny round ears.",
    "bird": "Bird villager: small rounded beak, tiny wings instead of arms, short legs, no feather detail.",
    "other": "Villager: rounded simplified animal form, oversized eyes, stubby limbs.",
}

# Stage 2: environment + rendering
AC_PIPELINE_ENVIRONMENT = (
    "Environment: random Animal Crossing island background, bright grass, stone or dirt path, fruit trees, "
    "cozy villager house, flower patches, soft Nintendo daytime lighting. "
    "Rendering: polished 3D Nintendo game render, soft global illumination, subtle ambient occlusion, "
    "smooth plastic-like shading, no painterly effect, no 2D illustration look, game-ready model, high resolution."
)

# Stage 2 & 4 negative (failsafe: reject realistic proportions / original background)
NEGATIVE_AC_PIPELINE = (
    "realistic animal proportions, photorealistic, original background, 2D flat background, "
    "realistic anatomy, realistic fur, long legs, long torso, painterly, low poly, blurry, "
    "asymmetrical face, floating character, incorrect shadow, dark cinematic lighting."
)


def build_ac_pipeline_base_prompt(species: str) -> str:
    """Stage 2: text prompt for base villager (T2I-like via high-strength img2img)."""
    key = species.lower().strip() if species else "other"
    if key not in AC_PIPELINE_SPECIES:
        key = "other"
    return (
        AC_PIPELINE_BASE_ANATOMY
        + AC_PIPELINE_SPECIES_ADAPT.get(key, AC_PIPELINE_SPECIES_ADAPT["other"])
        + " "
        + AC_PIPELINE_ENVIRONMENT
    )


def build_ac_pipeline_color_transfer_prompt(
    species: str,
    main_color: str,
    secondary_color: str,
    eye_color: str,
    markings: str,
) -> str:
    """Stage 4: same villager with only fur/eye/markings applied; proportions unchanged."""
    key = species.lower().strip() if species else "other"
    if key not in AC_PIPELINE_SPECIES:
        key = "other"
    anatomy = AC_PIPELINE_BASE_ANATOMY + AC_PIPELINE_SPECIES_ADAPT.get(key, AC_PIPELINE_SPECIES_ADAPT["other"])
    color_desc = f"Main fur color: {main_color or 'cream'}. Secondary fur: {secondary_color or 'none'}. Eye color: {eye_color or 'amber'}. Major markings: {markings or 'none'}."
    return (
        f"Same Animal Crossing villager, exact same proportions and pose. Apply only these colors: {color_desc} "
        f"Do not change head size, limb proportions, pose, background, lighting, or camera angle. "
        f"{anatomy} {AC_PIPELINE_ENVIRONMENT}"
    )


def get_random_ac_background(override: str | None = None) -> str:
    """동물의숲 배경 랜덤 선택. 실패 시 기본 배경 반환. override가 있으면 해당 문자열 사용."""
    if override and override.strip():
        return override.strip()
    try:
        return random.choice(AC_BACKGROUNDS)
    except Exception:
        return DEFAULT_AC_BACKGROUND


# 동물의숲 원본 보존 모드: species 표시명 (프롬프트용)
AC_SPECIES_DISPLAY: dict[str, str] = {
    "dog": "Dog",
    "cat": "Cat",
    "rabbit": "Bunny",
    "hamster": "Hamster",
    "bird": "Bird",
    "ferret": "Ferret",
    "turtle": "Turtle",
    "reptile": "Reptile",
    "pet": "Pet",
}

# 동물의숲 원본 보존: 참조 이미지 구성·배경·의상·포즈 최대한 유지, 반려동물만 AC 주민으로 변환 (마스터피스/바이브 코딩)
AC_PRESERVE_ORIGINAL_TEMPLATE = (
    "((A masterpiece quality, highly detailed 3D render of a charming {{SPECIES}} villager in the style of Animal Crossing: New Horizons)). "
    "This villager, a friendly {{SPECIES}} with a simplified, chibified form, is based on the specific pet in the reference image, capturing its proportions and heartwarming expression. "
    "It {{POSE}} on the exact same environment from the reference image, surrounded by the same lush scenery. "
    "The character is wearing the same clothing as in the reference image. "
    "Large, soulful, expressive {{EYE_COLOR}} eyes are wide and friendly, looking directly forward. "
    "The complex, charming background from the reference image is perfectly preserved{{SIGN_TEXT}}. "
    "Soft, warm, golden sunlight bathes the entire wholesome scene, casting inviting, realistic shadows and making the textures (knitted sweater, soft fur, smooth surfaces, blooming flowers) pop with crisp, clean lines and incredible depth. "
    "Cozy, delightful, charming atmosphere. The scene is rendered in an immersive 3D style, free from any distortion or artifacts, appearing as a high-resolution game asset. "
    "0.5:1 chibi proportions, head large and round, short limbs, soft touchable fur texture."
)

# 동물의숲 3D 스타일: species별 특징 (지시서와 동일)
ANIMAL_CROSSING_SPECIES: dict[str, str] = {
    "dog": "dog villager, rounded ears, short snout, canine 3D character, clearly a dog",
    "cat": "cat villager, pointed triangular ears, slit pupils, small nose no long snout, feline 3D character, clearly a cat",
    "rabbit": "rabbit villager, long ears, fluffy tail, round body, clearly a rabbit",
    "hamster": "hamster villager, plump cheeks, small round ears, round compact body, clearly a hamster",
    "bird": "bird villager, colorful feathers, beak and wing shapes, clearly a bird",
    "ferret": "ferret villager, elongated body, rounded ears",
    "turtle": "turtle villager, dome shell, simplified limbs",
    "reptile": "reptile villager, simplified 3D character",
    "pet": "",
}

# 픽셀 아트에서 종 구분을 확실히 하기 위한 스프라이트 힌트 (고양이/강아지 형태 명시)
PIXEL_ART_SPECIES_SPRITE: dict[str, str] = {
    "dog": "dog sprite with rounded or floppy ear blocks, short blunt muzzle block, canine silhouette",
    "cat": (
        "cat sprite with pointed triangular ear blocks, both ears with pink inner ear visible symmetrical, "
        "no long muzzle, small nose, feline silhouette"
    ),
    "rabbit": "rabbit sprite with long upright ear blocks, round body",
    "hamster": "hamster sprite with small round ears, round body",
    "ferret": "ferret sprite with rounded ears, elongated body",
    "bird": "bird sprite with beak wedge, wing shapes",
    "turtle": "turtle sprite with dome shell block",
    "reptile": "reptile sprite with simplified head and body",
    "pet": "",
}

NEGATIVE_BY_STYLE: dict[str, str] = {
    "dragonball": "soft shading, gradient, realistic muscle, photorealistic",
    "slamdunk": "soft shading, gradient, 3D, photorealistic",
    "sailor_moon": "dark gritty, gradient, photorealistic",
    "pokemon": "realistic fur, realistic feathers, gradient, photorealistic",
    "dooly": "complex shading, gradient, detailed texture",
    "mazinger": "organic texture, fur, gradient, photorealistic",
    "shinchan": "realistic, gradient, detailed anatomy",
    "pixel_art": (
        "gradient, anti-aliasing, smooth edges, high resolution, blur, soft shadow, 3D, voxel, "
        "edge artifact, chromatic aberration, muddy, desaturated strip, color bleed, noisy corners, "
        "ambiguous silhouette, generic indistinguishable animal"
    ),
    "pixel art": (
        "gradient, anti-aliasing, smooth edges, high resolution, blur, soft shadow, 3D, voxel, "
        "edge artifact, chromatic aberration, muddy, desaturated strip, color bleed, noisy corners, "
        "ambiguous silhouette, generic indistinguishable animal"
    ),
    "animal_crossing": (
        "2D flat background, hand-drawn textures, low-poly, blurry, realistic photorealism, original background, "
        "asymmetrical ears, human-like anatomy, color bleed, distorted proportions, dark or dull colors, pixelated artifacts"
    ),
    "animal crossing": (
        "2D flat background, hand-drawn textures, low-poly, blurry, realistic photorealism, original background, "
        "asymmetrical ears, human-like anatomy, color bleed, distorted proportions, dark or dull colors, pixelated artifacts"
    ),
    "animal_crossing_hybrid": (
        "flat 2D background, original background, photo background, realistic anatomy, real cat proportions, "
        "long legs, long torso, realistic fur texture, photorealism, low poly, blurry, asymmetrical face, "
        "floating character, incorrect ground shadow, dark cinematic lighting, overly dramatic shadows, hyper detailed whiskers"
    ),
    "animal crossing hybrid": (
        "flat 2D background, original background, photo background, realistic anatomy, real cat proportions, "
        "long legs, long torso, realistic fur texture, photorealism, low poly, blurry, asymmetrical face, "
        "floating character, incorrect ground shadow, dark cinematic lighting, overly dramatic shadows, hyper detailed whiskers"
    ),
}

# 동물의숲 원본 보존 모드 전용 네거티브 (배경/의상 변경 금지, 원본 유지 유도)
NEGATIVE_AC_PRESERVE = (
    "2D flat background, hand-drawn textures, low-poly, blurry, realistic photorealism, "
    "asymmetrical ears, human-like anatomy, color bleed, distorted proportions, "
    "different background from reference, changed clothing, different pose from reference, altered composition"
)

# ========== GENERATION RULES ==========
# pixel_art: 384 생성 후 512로 리사이즈(nearest). steps/guidance 상향으로 품질·종 구분 강화.
GENERATION_RULES: dict[str, dict[str, Any]] = {
    "pixel_art": {"max_side": 384, "steps": 48, "guidance_scale": 7.5},
    "pixel art": {"max_side": 384, "steps": 48, "guidance_scale": 7.5},
    "dragonball": {"max_side": 768, "steps": 30, "guidance_scale": 6.5},
    "slamdunk": {"max_side": 768, "steps": 30, "guidance_scale": 6.5},
    "sailor_moon": {"max_side": 768, "steps": 30, "guidance_scale": 6.5},
    "pokemon": {"max_side": 768, "steps": 30, "guidance_scale": 6.5},
    "dooly": {"max_side": 768, "steps": 28, "guidance_scale": 6.5},
    "mazinger": {"max_side": 768, "steps": 30, "guidance_scale": 7.0},
    "shinchan": {"max_side": 768, "steps": 28, "guidance_scale": 6.5},
    # 동물의숲 단일 스타일: strength 0.78, guidance 8.5, steps 48, 768x768, pose preservation OFF
    "animal_crossing": {"max_side": 768, "steps": 48, "guidance_scale": 8.5},
    "animal crossing": {"max_side": 768, "steps": 48, "guidance_scale": 8.5},
    "animal_crossing_hybrid": {"max_side": 768, "steps": 44, "guidance_scale": 8.0},
    "animal crossing hybrid": {"max_side": 768, "steps": 44, "guidance_scale": 8.0},
}

STYLE_TEMPLATES = STYLE_PROMPTS

# 스타일 목록: 동물의숲은 animal_crossing 하나만 노출 (animal crossing / hybrid는 API 호환용으로만 유지)
ALLOWED_STYLE_KEYS = [
    "dragonball",
    "slamdunk",
    "sailor_moon",
    "pokemon",
    "dooly",
    "mazinger",
    "shinchan",
    "pixel_art",
    "pixel art",
    "animal_crossing",
]

ALLOWED_SPECIES_KEYS = list(SPECIES_MODIFIERS.keys())


def _normalize_style(style: str | None) -> str | None:
    if style is None or not str(style).strip():
        return None
    key = str(style).strip().lower().replace(" ", "_")
    if key in STYLE_PROMPTS:
        return key
    if key.replace("_", " ") in STYLE_PROMPTS:
        return key.replace("_", " ")
    for k in STYLE_PROMPTS:
        if k.replace(" ", "_") == key:
            return k
    return style.strip().lower().replace(" ", "_")


def _normalize_species(species: str | None) -> str | None:
    if species is None or not str(species).strip():
        return None
    key = str(species).strip().lower().replace(" ", "_")
    return key if key in SPECIES_MODIFIERS else ("pet" if key else None)


def get_prompt_config() -> dict[str, Any]:
    """Full structured config: base_prompt, base_negative, species_modifiers, styles, generation_rules."""
    return {
        "base_prompt": BASE_PROMPT,
        "base_negative": BASE_NEGATIVE,
        "species_modifiers": dict(SPECIES_MODIFIERS),
        "styles": dict(STYLE_PROMPTS),
        "generation_rules": dict(GENERATION_RULES),
    }


def get_generation_rules(style_key: str | None) -> dict[str, Any]:
    """Get max_side, steps, guidance_scale for a style. pixel_art: 384, 48, 7.5; else 768, 28-30, 6.5-7."""
    default = {"max_side": 768, "steps": 30, "guidance_scale": 6.5}
    if not style_key:
        return default
    key = _normalize_style(style_key)
    if not key:
        return default
    return GENERATION_RULES.get(key, default)


# Default subject when user leaves prompt empty (generic small pet animal)
DEFAULT_USER_PROMPT = "small pet animal character"


def _build_ac_preserve_prompt(
    species_key: str | None,
    ac_eye_color: str | None,
    ac_pose: str | None,
    ac_sign_text: str | None,
) -> str:
    """동물의숲 원본 보존 모드: 참조 이미지 구성·배경·의상·포즈 유지, 반려동물만 AC 주민으로."""
    species_display = (
        AC_SPECIES_DISPLAY.get(species_key, "pet").capitalize()
        if species_key
        else "Pet"
    )
    eye = (ac_eye_color or "warm soulful eyes").strip()
    if eye and "eye" not in eye.lower():
        eye = f"{eye} eyes"
    pose = (ac_pose or "stands naturally, preserving the exact pose and composition from the reference image").strip()
    sign = ""
    if ac_sign_text and ac_sign_text.strip():
        sign = f', including a custom wooden town sign that reads "{ac_sign_text.strip()}"'
    return AC_PRESERVE_ORIGINAL_TEMPLATE.replace("{{SPECIES}}", species_display).replace(
        "{{EYE_COLOR}}", eye
    ).replace("{{POSE}}", pose).replace("{{SIGN_TEXT}}", sign)


def build_prompt(
    user_prompt: str,
    style: str | None = None,
    species: str | None = None,
    raw_prompt: bool = False,
    ac_background: str | None = None,
    ac_preserve_original: bool = False,
    ac_eye_color: str | None = None,
    ac_pose: str | None = None,
    ac_sign_text: str | None = None,
) -> str:
    """
    Final prompt = (종 주어) + user_prompt + BASE_PROMPT + species_modifier + style rules.
    animal_crossing + ac_preserve_original 이면 원본 보존 템플릿 사용(참조 이미지 구성·배경·의상·포즈 유지).
    """
    text = (user_prompt or "").strip()
    species_key = _normalize_species(species)
    style_key = _normalize_style(style)
    is_animal_crossing = style_key in ("animal_crossing", "animal crossing")
    is_animal_crossing_hybrid = style_key in ("animal_crossing_hybrid", "animal crossing hybrid")

    # 동물의숲 원본 보존 모드: 단일 마스터피스 프롬프트 사용 (hybrid는 고정 마을 배경 사용)
    if is_animal_crossing and not is_animal_crossing_hybrid and ac_preserve_original:
        return _build_ac_preserve_prompt(
            species_key, ac_eye_color, ac_pose, ac_sign_text
        )

    if not raw_prompt and species_key and species_key in SPECIES_SUBJECT:
        subject = SPECIES_SUBJECT[species_key]
        if not text:
            text = subject
        else:
            text = f"{subject}, {text}"
    elif not text:
        text = DEFAULT_USER_PROMPT
    if raw_prompt:
        return text
    parts = [text, BASE_PROMPT]
    # 동물의숲: 완전 재디자인 모드 → 포즈/원본 구도 보존 안 함 (pose preservation OFF)
    if not (is_animal_crossing or is_animal_crossing_hybrid):
        parts.append(POSE_PRESERVATION)
    if species_key and species_key in SPECIES_MODIFIERS:
        parts.append(SPECIES_MODIFIERS[species_key])
    if style_key and style_key in STYLE_PROMPTS:
        style_part = STYLE_PROMPTS[style_key]
        if "{{AC_BACKGROUND}}" in style_part:
            style_part = style_part.replace(
                "{{AC_BACKGROUND}}",
                get_random_ac_background(ac_background),
            )
        parts.append(style_part)
    if (is_animal_crossing or is_animal_crossing_hybrid) and species_key and species_key in ANIMAL_CROSSING_SPECIES and ANIMAL_CROSSING_SPECIES[species_key]:
        parts.append(ANIMAL_CROSSING_SPECIES[species_key])
    is_pixel_art = style_key in ("pixel_art", "pixel art")
    if is_pixel_art and species_key and species_key in PIXEL_ART_SPECIES_SPRITE and PIXEL_ART_SPECIES_SPRITE[species_key]:
        parts.append(PIXEL_ART_SPECIES_SPRITE[species_key])
        if species_key == "dog":
            parts.append("clearly a dog, not a cat")
        elif species_key == "cat":
            parts.append("clearly a cat, not a dog")
    return ", ".join(parts)


def build_negative_prompt(
    style: str | None,
    species: str | None = None,
    raw_prompt: bool = False,
    ac_preserve_original: bool = False,
) -> str:
    """Final negative = BASE_NEGATIVE + pose-avoid + species cross-avoid + style-specific negative."""
    if raw_prompt:
        return ""
    parts = [BASE_NEGATIVE, POSE_NEGATIVE]
    species_key = _normalize_species(species)
    if species_key and species_key in SPECIES_NEGATIVE_AVOID and SPECIES_NEGATIVE_AVOID[species_key]:
        parts.append(SPECIES_NEGATIVE_AVOID[species_key])
    style_key = _normalize_style(style)
    is_animal_crossing = style_key in ("animal_crossing", "animal crossing")
    if is_animal_crossing and ac_preserve_original:
        parts.append(NEGATIVE_AC_PRESERVE)
    else:
        style_neg = NEGATIVE_BY_STYLE.get(style_key, "") if style_key else ""
        if style_neg:
            parts.append(style_neg)
    return ", ".join(p for p in parts if p)


def get_allowed_style_keys() -> list[str]:
    """API: allowed style keys (no duplicate display)."""
    seen: set[str] = set()
    out: list[str] = []
    for k in ALLOWED_STYLE_KEYS:
        if k not in seen:
            seen.add(k)
            out.append(k)
    return out


def get_allowed_species_keys() -> list[str]:
    """API: allowed species keys for species_modifiers."""
    return list(ALLOWED_SPECIES_KEYS)
