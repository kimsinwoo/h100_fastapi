"""
이미지 생성 AI 변수 통제 및 퀄리티 극대화.
스타일별 고유 특성 강화: 픽셀아트=순수 2D만, 나머지=카테고리별 정확한 미디엄.
"""

from __future__ import annotations


class ImagePromptExpert:
    """
    스타일 고유성 확보: 각 카테고리만의 특성이 나오도록 Positive/Negative 분리 강화.
    """

    # 스타일 앵커(맨 앞) + 고유 키워드로 카테고리 구분
    STYLE_POS_DB: dict[str, str] = {
        "pixel art": (
            "2D pixel art only, flat image like a PNG sprite from a 2D game. "
            "Single flat plane, like Celeste or Stardew Valley character sprite, not 3D. "
            "NOT voxel, NOT Minecraft, NOT 3D blocks, NOT isometric, NOT lego style. "
            "(({subject})), drawn with square pixels on a flat grid, hard edges, "
            "limited palette, bold outlines, flat solid background, front or side view only, "
            "zero depth zero volume, no rounded blocks, no sculpted look, pixel-perfect 2D sprite"
        ),
        "realistic": (
            "REAL PHOTOGRAPH. Not drawn not painted. "
            "((Hyper-photorealistic)), {subject}, "
            "shot on Phase One 150MP, 80mm lens f/8, Hasselblad color, "
            "skin pores, vellus hair, natural light, RAW, National Geographic quality, "
            "no painting, no cartoon, no illustration, no 3D render"
        ),
        "anime": (
            "JAPANESE ANIME. Cel-shaded 2D only. "
            "((High-budget anime key visual)), {subject}, "
            "flat cel shading, sharp black line art, no gradients on skin, "
            "large anime eyes with highlights, Kyoto Animation / Makoto Shinkai style, "
            "vibrant colors, hand-drawn look, no 3D, no CGI, no photorealism"
        ),
        "3d render": (
            "3D CGI RENDER. Not 2D not photo. "
            "((Octane / Unreal Engine 5 render)), {subject}, "
            "path tracing, PBR materials, subsurface scattering, volumetric lighting, "
            "high-poly smooth mesh, ray tracing, no 2D art, no photograph"
        ),
        "cinematic": (
            "MOVIE FRAME. Film still. "
            "((Hollywood cinema still)), {subject}, "
            "2.39:1 anamorphic, film grain, teal and orange grading, "
            "depth of field, rim light, Roger Deakins style, "
            "no cartoon, no anime, no 2D illustration"
        ),
        "watercolor": (
            "TRADITIONAL WATERCOLOR ON PAPER. "
            "((Hand-painted watercolor)), {subject}, "
            "Arches paper texture, wet-on-wet, pigment bleeding, transparent washes, "
            "no digital lines, no solid fills, no photograph, no oil paint"
        ),
        "oil painting": (
            "OIL PAINT ON CANVAS. "
            "((Museum oil painting)), {subject}, "
            "impasto brushstrokes, canvas texture, Chiaroscuro, Rembrandt style, "
            "no digital, no photo, no anime, no flat cel shading"
        ),
        "sketch": (
            "PENCIL SKETCH ON PAPER. Monochrome. "
            "((Graphite charcoal drawing)), {subject}, "
            "cross-hatching, HB 2B pencil, Canson paper, high contrast black and white, "
            "no color, no paint, no 3D, no photograph"
        ),
        "cyberpunk": (
            "CYBERPUNK NEON NIGHT. "
            "((Neon noir futuristic)), {subject}, "
            "blue magenta neon, wet streets, volumetric fog, holographic UI, "
            "no daylight, no nature, no pastel, no vintage"
        ),
        "fantasy art": (
            "FANTASY ILLUSTRATION. Medieval magical. "
            "((Epic fantasy digital art)), {subject}, "
            "Greg Rutkowski style, ornate armor, glowing particles, dragons castles, "
            "no modern, no sci-fi, no photograph, no pixel art"
        ),
    }

    STYLE_NEG_DB: dict[str, str] = {
        "pixel art": (
            "voxel, voxel art, 3D blocks, Minecraft, lego, cubes, blocky 3D, "
            "isometric, isometric 3D, perspective, depth, volume, thickness, "
            "sculpted, rounded, smooth shading, soft edges, anti-aliasing, "
            "ray tracing, CGI, 3D render, photorealism, realistic lighting, "
            "shadow depth, gradient, volumetric, rectangular blocks, "
            "striped sweater made of blocks, 3D model, Unreal Engine, Unity"
        ),
        "realistic": (
            "drawing, painting, cartoon, anime, illustration, 3D render, CGI, "
            "stylized, plastic skin, airbrushed, cel shading, watercolor, oil, sketch, pixel art"
        ),
        "anime": (
            "3D, CGI, photorealistic, photograph, western cartoon, "
            "gradient on skin, airbrushed, soft shading, clay, messy lines, realistic eyes"
        ),
        "3d render": (
            "2D, flat, sketch, painting, pixel art, hand-drawn, watercolor, oil, photograph, anime"
        ),
        "cinematic": (
            "cartoon, anime, 2D illustration, drawing, painting, flat lighting, "
            "amateur, phone camera, pixel art, watercolor, sketch"
        ),
        "watercolor": (
            "digital lines, vector, solid fills, 3D, photograph, oil, acrylic, "
            "neon, gradient, pixel art, anime, CGI"
        ),
        "oil painting": (
            "flat digital, smooth gradient, anime, vector, photograph, cel shading, pixel art, sketch, watercolor"
        ),
        "sketch": (
            "color, painted, digital paint, 3D render, photograph, watercolor, oil, neon, cel shading, pixel art"
        ),
        "cyberpunk": (
            "daylight, sunny, nature, forest, rustic, vintage, pastel, soft lighting, "
            "watercolor, oil painting, anime, cartoon"
        ),
        "fantasy art": (
            "modern, photograph, sci-fi, technology, pixel art, 3D voxel, Minecraft, minimalist, cartoon"
        ),
    }

    BASE_NEGATIVE = (
        "lowres, bad anatomy, bad hands, text, error, missing fingers, extra digit, "
        "cropped, worst quality, low quality, jpeg artifacts, signature, watermark, "
        "blurry, distorted, deformed, extra limbs, duplicate, mutilated"
    )

    DEFAULT_STYLE = "realistic"

    @classmethod
    def _get_style_key(cls, style: str) -> str:
        key = style.lower().strip()
        if key in cls.STYLE_POS_DB:
            return key
        key_no_space = key.replace(" ", "")
        for k in cls.STYLE_POS_DB:
            if k.replace(" ", "") == key_no_space:
                return k
        return cls.DEFAULT_STYLE

    @classmethod
    def compile(
        cls,
        style: str,
        user_content: str | None,
        aspect_ratio: str = "1:1",
    ) -> dict[str, str]:
        style_key = cls._get_style_key(style)
        pos_template = cls.STYLE_POS_DB.get(style_key, cls.STYLE_POS_DB[cls.DEFAULT_STYLE])
        neg_template = cls.STYLE_NEG_DB.get(
            style_key, cls.STYLE_NEG_DB.get(cls.DEFAULT_STYLE, "")
        )

        user = (user_content or "").strip()
        content_emphasized = f"((({user})))" if user else "((detailed subject, high quality))"

        final_positive = pos_template.format(subject=content_emphasized)
        final_positive += f", masterpiece, 8k, aspect ratio {aspect_ratio}"

        final_negative = f"{cls.BASE_NEGATIVE}, {neg_template}"

        return {
            "style": style_key,
            "final_prompt": final_positive,
            "negative_prompt": final_negative,
            "consistency_hint": "To maintain character, use Gen ID and Seed from previous result.",
        }
