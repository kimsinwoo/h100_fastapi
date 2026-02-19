"""
Expert Image Prompt Engineer — 최대 강화 스타일별 Positive/Negative.
"""

from __future__ import annotations

# ========== Positive: 최대 강화, 중복 강조 포함 ==========
STYLE_PROMPTS: dict[str, str] = {
    "anime": (
        "masterpiece best quality 2D anime only, flat cel shading absolutely no gradients on skin or hair, "
        "clean crisp black outline line art, vibrant saturated colors, large expressive anime eyes with multiple highlights and sparkle, "
        "hand-drawn traditional anime look, Makoto Shinkai or Studio Ghibli or Kyoto Animation aesthetic, "
        "4k 8k resolution, sharp crisp edges, 2D flat only, no 3D ever, no CGI, no photorealistic, no western cartoon, "
        "detailed hair strands, soft cel shadows, professional anime key visual"
    ),
    "realistic": (
        "masterpiece best quality photorealistic photograph, 8k UHD ultra detailed, shot on 35mm lens f/1.8 aperture bokeh, "
        "RAW uncompressed format, Fujifilm or Canon color science, visible skin pores texture wrinkles fine detail, "
        "individual hair strands eyelashes, natural daylight or soft studio lighting, tack sharp focus, "
        "professional photography, National Geographic quality, absolutely no painting no drawing no cartoon no illustration no 3D render"
    ),
    "watercolor": (
        "masterpiece authentic traditional watercolor painting on cold-press or rough paper, wet-on-wet technique, "
        "visible paper texture and fiber, soft bleeding color edges and blooms, paint drip runs and backruns, "
        "hand-painted brush strokes visible, transparent layered washes, organic irregular edges, "
        "delicate pigment granulation, no digital art, no sharp vector lines, no solid flat fills, no photograph, no 3D"
    ),
    "cyberpunk": (
        "masterpiece high-contrast cyberpunk neon aesthetic, electric blue cyan and hot magenta pink lighting, "
        "night city scene, wet asphalt pavement reflections, rain droplets, volumetric light beams and fog, "
        "smog smoke haze, deep shadows and bright neon, sharp focus, holographic displays, high-tech machinery, "
        "Blade Runner or Ghost in the Shell atmosphere, absolutely no daylight no sun no nature no vintage no pastel no rustic"
    ),
    "oil painting": (
        "masterpiece classical oil painting on canvas, heavy impasto thick visible brushstrokes, "
        "visible canvas weave texture, built-up paint layers, Chiaroscuro dramatic light and shadow, "
        "Renaissance or Baroque or Rembrandt influence, rich oil pigments, museum gallery masterpiece quality, "
        "traditional fine art, absolutely no flat digital no thin wash no photograph no anime no vector no cel shading"
    ),
    "sketch": (
        "masterpiece graphite pencil or charcoal drawing on white textured drawing paper, hand-drawn only, "
        "cross-hatching stippling and contour lines for shading, rough unfinished artistic sketch lines, "
        "high contrast black and white only, minimalist composition, HB 2B 4B pencil grain texture visible, "
        "figure study or concept sketch style, absolutely no color no paint no 3D no photograph no digital no vector"
    ),
    "cinematic": (
        "masterpiece single frame extracted from Hollywood movie, anamorphic lens flare and bokeh, "
        "2.35:1 or 2.39:1 widescreen cinema aspect ratio, teal and orange professional color grading, "
        "shallow depth of field, dramatic key light and rim light, film grain texture, blockbuster still, "
        "Roger Deakins or Emmanuel Lubezki cinematography style, absolutely no cartoon no anime no flat lighting no phone photo no 2D art"
    ),
    "fantasy art": (
        "masterpiece epic fantasy digital illustration, extremely intricate detailed armor fabric jewelry and weapons, "
        "magical glow, floating particles, luminous ethereal atmosphere, dragons castles knights wizards, "
        "Greg Rutkowski or Artgerm or WLOP style, heroic dynamic pose, dramatic composition, 8k high resolution, "
        "concept art quality, absolutely no modern technology no sci-fi no photograph no pixel art no 3D voxel no low detail"
    ),
    "pixel art": (
        "masterpiece strictly 2D pixel art only, completely flat single plane, zero depth zero perspective zero volume, "
        "16-bit SNES Sega Genesis era sprite style, square blocky pixels only, no anti-aliasing, hard aliased jagged edges, "
        "single character or object sprite, non-isometric, side view or front view only never isometric, "
        "maximum 8 to 16 colors limited palette, bold thick black outline around every shape, "
        "retro video game character sprite sheet, absolutely no Minecraft no voxel no 3D blocks no lego no isometric no round no sculpted no depth no volume no 3D render no CGI"
    ),
    "3d render": (
        "masterpiece Octane Render or Unreal Engine 5 or V-Ray, path-traced ray tracing, 8k resolution output, "
        "volumetric god rays and light scattering, subsurface scattering on skin, PBR physically based materials, "
        "high-polygon smooth subdivision surface, hyper-realistic detailed, studio HDRI environment lighting, "
        "photorealistic CGI, absolutely no 2D no flat illustration no sketch no painting no pixel art no hand-drawn no cartoon"
    ),
}

# ========== Negative: 최대 강화, 배제 항목 대폭 확대 ==========
STYLE_NEGATIVE_PROMPTS: dict[str, str] = {
    "anime": (
        "3D render, CGI, photorealistic, photograph, realistic, blurry, grainy, noisy, lowres, low resolution, "
        "messy lines, sketchy, 3D model, plastic look, gradient hair, western cartoon, realistic eyes, realistic skin, "
        "clay render, voxel, Minecraft, isometric 3D, depth map, normal map, PBR, ray tracing, "
        "oil painting, watercolor, pixel art, vector art, ugly, duplicate, morbid, mutilated, extra limbs"
    ),
    "realistic": (
        "drawing, painting, cartoon, anime, illustration, 3D render, CGI, stylized, artistic, "
        "plastic skin, airbrushed, doll eyes, deformed anatomy, distorted limbs, oversaturated, "
        "painted texture, cartoon outline, cel shading, watercolor, oil painting, sketch, pixel art, "
        "lowres, blurry, ugly, duplicate, morbid, mutilated, extra limbs, bad anatomy"
    ),
    "watercolor": (
        "sharp digital lines, vector, solid flat fills, 3D, photograph, photorealistic, "
        "oil painting, acrylic, plastic surface, neon colors, digital gradient, clean cut edges, "
        "pixel art, anime, cel shading, ray tracing, CGI, lowres, blurry, ugly, "
        "thick paint, impasto, canvas texture"
    ),
    "cyberpunk": (
        "bright daylight, sunny, sun, natural light, forest, countryside, nature, trees, grass, "
        "rustic, vintage, retro, pastel colors, soft lighting, low contrast, flat lighting, "
        "simplistic, minimal, daytime, golden hour, warm tone, peaceful, pastoral, "
        "watercolor, oil painting, anime, cartoon, pixel art, lowres, blurry"
    ),
    "oil painting": (
        "flat color, smooth gradient, digital art, anime, vector, 2D illustration, clean lines, "
        "thin wash, photograph, photo, cel shading, pixel art, sketch, watercolor, "
        "3D render, CGI, plastic, neon, lowres, blurry, ugly, duplicate, "
        "modern art, minimalist, abstract"
    ),
    "sketch": (
        "full color, colored, painted, digital paint, 3D render, smooth gradient, "
        "clean vector lines, photograph, photo, blurry, polished, finished painting, "
        "ink, watercolor, oil, neon, saturated, cel shading, pixel art, "
        "lowres, ugly, duplicate, anime, cartoon, realistic"
    ),
    "cinematic": (
        "cartoon, anime, flat lighting, amateur, snapshot, phone camera, smartphone, "
        "bright cheerful, low contrast, 2D illustration, drawing, painting, "
        "overexposed, underexposed, no depth of field, flat depth, "
        "pixel art, watercolor, oil painting, sketch, lowres, blurry, ugly"
    ),
    "fantasy art": (
        "modern clothing, modern setting, photograph, photo, sci-fi, technology, spaceship, "
        "pixel art, low detail, simple background, empty background, 3D voxel, Minecraft, "
        "realistic portrait, minimalist, cartoon, anime, lowres, blurry, ugly, "
        "flat colors, vector, cel shading"
    ),
    "pixel art": (
        "Minecraft, voxel, 3D blocks, lego, blocks, cubes, isometric 3D, isometric view, "
        "depth, volume, perspective, round, sculpted, curved, smooth shading, anti-aliased, "
        "gradient, soft shadow, photorealistic, 3D render, CGI, ray tracing, PBR, "
        "realistic, photograph, smooth edges, blur, depth of field, bokeh, "
        "oil painting, watercolor, anime, cartoon, lowres, ugly, duplicate"
    ),
    "3d render": (
        "2D, flat illustration, sketch, painting, pixel art, hand-drawn, "
        "low-poly, low poly, faceted, grainy, cartoon outline, cel shading, "
        "watercolor, oil painting, photograph, photo, anime, "
        "lowres, blurry, ugly, duplicate, bad geometry"
    ),
}

BASE_NEGATIVE = (
    "blurry, low quality, distorted, watermark, text, signature, logo, "
    "ugly, duplicate, morbid, mutilated, poorly drawn, bad anatomy, wrong anatomy, extra limbs"
)

DEFAULT_STYLE = "realistic"

STYLE_PRESETS: dict[str, str] = {
    "anime": "Anime (2D Cel-Shaded)",
    "realistic": "Realistic (Photographic)",
    "watercolor": "Watercolor (Traditional)",
    "cyberpunk": "Cyberpunk (Futuristic)",
    "oil painting": "Oil Painting (Classical)",
    "sketch": "Sketch (Hand-drawn)",
    "cinematic": "Cinematic (Movie Scene)",
    "fantasy art": "Fantasy Art (Digital Illustration)",
    "pixel art": "Pixel Art (2D Flat Sprite)",
    "3d render": "3D Render (CGI)",
}


def get_style_prompt(style_key: str) -> str:
    key = style_key.strip().lower()
    return STYLE_PROMPTS.get(key, STYLE_PROMPTS[DEFAULT_STYLE])


def get_style_negative_prompt(style_key: str) -> str:
    key = style_key.strip().lower()
    neg = STYLE_NEGATIVE_PROMPTS.get(key, STYLE_NEGATIVE_PROMPTS[DEFAULT_STYLE])
    return f"{BASE_NEGATIVE}, {neg}"


def merge_prompt(style_key: str, custom_prompt: str | None) -> str:
    """Final Prompt = (User Content first), (Style Positive)."""
    style_text = get_style_prompt(style_key)
    if not (custom_prompt and custom_prompt.strip()):
        return style_text
    return f"{custom_prompt.strip()}, {style_text}"
