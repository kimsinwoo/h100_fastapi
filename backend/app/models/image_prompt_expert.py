"""
이미지 생성 AI 변수 통제 및 퀄리티 극대화 프롬프트 엔지니어링 모듈.
학습된 8대 구성 요소(Medium, Subject, Setting, Lighting, Color, Mood, Composition, Ratio) 반영.
"""

from __future__ import annotations


class ImagePromptExpert:
    """
    이미지 생성 AI 변수 통제 및 퀄리티 극대화 프롬프트 엔지니어링 모듈.
    학습된 8대 구성 요소(Medium, Subject, Setting, Lighting, Color, Mood, Composition, Ratio) 반영.
    """

    # 1. POSITIVE: 광학·물리·예술 기법 사양 (강조 순서 최적화), {subject}에 사용자 주제 삽입
    STYLE_POS_DB: dict[str, str] = {
        "pixel art": (
            "((Strictly 2D flat pixel art)), (16-bit GBA SNES hardware style:1.3), {subject}, "
            "perfect grid alignment, integer scaling, zero sub-pixels, hand-placed pixels, "
            "strictly flat single 2D plane, zero depth, zero perspective, no Z-axis, no 3D shadows, "
            "bold 2px black outline, high-contrast limited color palette max 32 colors, "
            "sharp aliased jagged edges, retro game sprite sheet, pure 2D illustration, "
            "strictly non-isometric, side-scroller aesthetic, no parallax, "
            "no anti-aliasing, no interpolation, vintage CRT scanline-ready"
        ),
        "realistic": (
            "((Hyper-photorealistic 16k resolution)), (Shot on Phase One XF IQ4 150MP:1.4), {subject}, "
            "Schneider Kreuznach 80mm LS lens, f/8.0 aperture, ISO 100, Hasselblad color science, "
            "unmatched skin micro-texture, visible pores, vellus hair, moisture on eyes, "
            "ray-traced reflections, subsurface scattering (SSS), natural sunlight 5500K, "
            "diffuse sky radiation, zero digital noise, National Geographic award-winning photography, "
            "RAW uncompressed format, 32-bit floating point color, hyper-detailed textures"
        ),
        "anime": (
            "((Masterpiece ultra-detailed 2D anime)), (Strictly flat cel shading:1.3), {subject}, "
            "official art style, high-budget 4K anime movie, sharp ink lines, zero gradients on skin or hair, "
            "Kyoto Animation lighting physics, vivid cinematic color grading, intricate eye refraction, "
            "8k resolution, line art perfection, hand-drawn 2D aesthetic, no 3D shading, no depth blur, "
            "no CGI artifacts, crisp edges, production I.G quality"
        ),
        "3d render": (
            "((Hyper-realistic 3D render)), (Unreal Engine 5.4.2 Lumen & Nanite:1.3), {subject}, "
            "Octane Render 2026, path-traced global illumination, ray-traced ambient occlusion, "
            "PBR materials 4.0, 8k texture maps, displacement mapping, subsurface scattering, "
            "volumetric god rays, Disney-Pixar production pipeline, flawless geometry, "
            "physically accurate light behavior, caustics, high-poly subdivision surface"
        ),
        "cinematic": (
            "((Hollywood IMAX 70mm film still)), (Shot on Panavision Millennium DXL2:1.3), {subject}, "
            "2.39:1 anamorphic widescreen, anamorphic bokeh, professional movie color grading, "
            "wide dynamic range HDR10, volumetric atmosphere, rim lighting, cinematic lens flares, "
            "Kodak Vision3 35mm film grain, Roger Deakins lighting style, filmic contrast, realistic light decay"
        ),
        "watercolor": (
            "((Authentic traditional watercolor)), (Arches 300gsm cold-press paper texture:1.3), {subject}, "
            "heavy pigment granulation, wet-on-wet technique, natural pigment sedimentation, "
            "delicate transparent washes, organic bleeding edges, hand-painted perfection, "
            "Winsor & Newton pigments, rough deckle edges, no digital smoothness, no vector lines"
        ),
        "oil painting": (
            "((Museum quality oil on linen canvas)), (Heavy impasto thick visible brushstrokes:1.3), {subject}, "
            "visible bristle marks, Rembrandt Chiaroscuro lighting, Baroque composition, "
            "rich mineral pigments, cracked varnish texture (craquelure), layered glazing, "
            "Old Masters fine art, traditional oil media only, high physical texture"
        ),
        "sketch": (
            "((Hyper-detailed graphite pencil drawing)), (Professional figure study:1.3), {subject}, "
            "masterful cross-hatching, 6B/9B pencil depth, Canson paper texture, "
            "hand-drawn artistic strokes, charcoal grit, high-contrast monochrome, "
            "Da Vinci sketchbook style, zero digital rendering, sharp HB outlines"
        ),
        "cyberpunk": (
            "((Hyper-detailed cyberpunk dystopia)), (Neon-saturated cinematic lighting:1.3), {subject}, "
            "Arri Alexa 65, ray-traced reflections on wet asphalt, volumetric fog, "
            "high-tech low-life, intricate mechanical detailing, holographic UI, teal and orange, "
            "8k resolution, industrial grit, maximum visual density"
        ),
        "fantasy art": (
            "((Transcendent fantasy illustration)), (Greg Rutkowski & Alphonse Mucha style:1.2), {subject}, "
            "intricate ornate filigree, ethereal glowing particles, magical aura, "
            "heroic dynamic composition, legendary artifact details, ArtStation trending, "
            "mythic atmosphere, 8k high-fidelity, vibrant magical colors"
        ),
    }

    # 2. NEGATIVE: 스타일별 오류·저품질 원천 차단 매트릭스
    STYLE_NEG_DB: dict[str, str] = {
        "pixel art": (
            "3D, 2.5D, voxel, minecraft, isometric, perspective, depth, volume, 3D blocks, lego, cubes, "
            "shadow gradients, soft edges, anti-aliasing, smooth shading, realistic lighting, ray tracing, "
            "render, CGI, blur, depth of field, bokeh, high-poly, rounded edges, sculpted, 3D model, "
            "unreal engine, unity, lowres, interpolation, smudged pixels, fuzzy edges, gradient fills, "
            "diagonal perspective, shadow depth, realistic textures, photorealism"
        ),
        "realistic": (
            "drawing, painting, cartoon, anime, illustration, 3D render, CGI, stylized, plastic skin, "
            "airbrushed, doll eyes, deformed anatomy, distorted limbs, oversaturated, painted texture, "
            "cartoon outlines, cel shading, watercolor, oil painting, sketch, pixel art, lowres"
        ),
        "anime": (
            "3D, CGI, render, realistic, photograph, gradient, airbrushed, soft shading, 3D model, "
            "clay, depth of field, blur, messy lines, sketchy, western cartoon, plastic texture, "
            "realistic skin, realistic eyes, heavy shadows, bad anatomy, lowres, text"
        ),
        "3d render": (
            "2D, flat illustration, sketch, painting, pixel art, hand-drawn, low-poly, grainy, "
            "cartoon outline, cel shading, watercolor, oil painting, photograph, anime, bad geometry"
        ),
        "cinematic": (
            "cartoon, anime, flat lighting, amateur, phone camera, bright cheerful, low contrast, "
            "2D illustration, drawing, painting, overexposed, underexposed, pixel art, watercolor, sketch"
        ),
        "watercolor": (
            "sharp digital lines, vector, solid flat fills, 3D, photograph, oil painting, acrylic, "
            "plastic surface, neon colors, digital gradient, clean cut edges, pixel art, anime, CGI"
        ),
        "oil painting": (
            "flat color, smooth gradient, digital art, anime, vector, 2D illustration, clean lines, "
            "thin wash, photograph, cel shading, pixel art, sketch, watercolor, 3D render"
        ),
        "sketch": (
            "full color, painted, digital paint, 3D render, smooth gradient, photograph, blurry, "
            "polished, ink, watercolor, oil, neon, saturated, cel shading, pixel art"
        ),
        "cyberpunk": (
            "bright daylight, sunny, nature, forest, rustic, vintage, pastel, soft lighting, "
            "low contrast, flat lighting, minimal, daytime, golden hour, watercolor, oil painting, anime"
        ),
        "fantasy art": (
            "modern clothing, modern setting, photograph, sci-fi, technology, pixel art, low detail, "
            "simple background, 3D voxel, Minecraft, realistic portrait, minimalist, cartoon"
        ),
    }

    # 범용 네거티브 (인체 오류 및 저품질 요소)
    BASE_NEGATIVE = (
        "lowres, bad anatomy, bad hands, text, error, missing fingers, extra digit, fewer digits, "
        "cropped, worst quality, low quality, normal quality, jpeg artifacts, signature, watermark, "
        "username, blurry, out of focus, distorted, deformed, mutated, extra limbs, fused fingers, "
        "cloning, duplicate, morbid, mutilated, bad proportions, gross proportions, long neck, "
        "unnatural body, double head, stacked people, floating limbs, disconnected limbs"
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
        """
        [학습 내용 반영] 주제(Subject)를 앞단에 배치하여 가중치를 높이고 스타일별 최적화 수행.
        user_content가 비어 있으면 기본 주제 문구로 대체.
        """
        style_key = cls._get_style_key(style)
        pos_template = cls.STYLE_POS_DB.get(style_key, cls.STYLE_POS_DB[cls.DEFAULT_STYLE])
        neg_template = cls.STYLE_NEG_DB.get(
            style_key, cls.STYLE_NEG_DB.get(cls.DEFAULT_STYLE, "")
        )

        user = (user_content or "").strip()
        content_emphasized = f"((({user})))" if user else "((detailed subject, high quality))"

        final_positive = pos_template.format(subject=content_emphasized)
        final_positive += f", masterpiece, extremely detailed, 8k, aspect ratio {aspect_ratio}"

        final_negative = f"{cls.BASE_NEGATIVE}, {neg_template}"

        return {
            "style": style_key,
            "final_prompt": final_positive,
            "negative_prompt": final_negative,
            "consistency_hint": "To maintain character, use Gen ID and Seed from previous result.",
        }
