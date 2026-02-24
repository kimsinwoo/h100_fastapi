from __future__ import annotations


class ImagePromptExpert:
    """
    최고급 이미지 품질 지향 프롬프트 설계.
    물리 기반 디테일 + 스타일 침범 차단 + diffusion artifact 억제.
    """

    DEFAULT_STYLE = "realistic"

    # ===== 공통 품질 강화 =====
    BASE_QUALITY = (
        "ultra-detailed, extreme micro-detail, "
        "physically accurate lighting, global illumination, "
        "high dynamic range, perfect texture fidelity, "
        "sharp natural focus, realistic contrast, "
        "professional color science, zero artificial smoothing"
    )

    BASE_NEGATIVE = (
        "lowres, blurry, soft focus, jpeg artifacts, compression artifacts, "
        "oversharpened, overprocessed, plastic skin, wax texture, ai artifacts, "
        "bad anatomy, bad hands, extra fingers, extra limbs, missing fingers, "
        "mutated, distorted, malformed, duplicate subject, "
        "text, watermark, logo, signature"
    )

    # ===== 스타일별 Positive =====
    STYLE_POS_DB: dict[str, str] = {

        "realistic": (
            "REAL PHOTOGRAPH. Not illustration. "
            "((Ultra photorealistic documentary photography)), {subject}, "
            "captured on Phase One XF IQ4 150MP, 80mm lens f/8, "
            "medium format sensor, natural lens compression, "
            "true skin microtexture, visible pores, vellus hair, "
            "subsurface scattering in skin, accurate specular highlights, "
            "soft natural window light, bounce light, "
            "RAW dynamic range, National Geographic cover quality"
        ),

        "cinematic": (
            "CINEMA FRAME. Feature film still. "
            "((Hollywood grade cinematography)), {subject}, "
            "shot on ARRI Alexa 65, anamorphic 2.39:1, "
            "motivated lighting design, rim light separation, "
            "controlled shadows, volumetric atmosphere, "
            "Kodak 2383 film emulation, subtle film grain, "
            "award-winning cinematography quality"
        ),

        "anime": (
            "JAPANESE THEATRICAL ANIME. Strict 2D only. "
            "((High-budget anime key visual)), {subject}, "
            "clean cel shading, precise black lineart, consistent line weight, "
            "two-tone shading only, no realistic lighting, "
            "flat color blocks, vibrant but controlled palette, "
            "hand-drawn look, zero 3D influence"
        ),

        "3d render": (
            "AAA CGI RENDER. Not photo not 2D. "
            "((Unreal Engine 5 cinematic render)), {subject}, "
            "Nanite geometry, Lumen global illumination, "
            "path traced lighting, physically based materials, "
            "accurate roughness map, subsurface scattering, "
            "ray-traced reflections, high polygon smooth topology"
        ),

        "pixel art": (
            "PURE 2D PIXEL ART SPRITE. "
            "((Flat 2D game sprite sheet style)), {subject}, "
            "exact square pixel grid alignment, "
            "limited 16-32 color palette, "
            "no gradient, no anti-aliasing, "
            "hard pixel edges, crisp silhouette, "
            "front or side view only, zero depth, zero volume"
        ),

        "watercolor": (
            "TRADITIONAL WATERCOLOR ON COTTON PAPER. "
            "((Hand-painted watercolor)), {subject}, "
            "Arches cold press texture, wet-on-wet technique, "
            "natural pigment bleeding, transparent washes, "
            "soft organic edges, no digital artifacts"
        ),

        "oil painting": (
            "OIL PAINT ON CANVAS. "
            "((Museum quality oil painting)), {subject}, "
            "impasto brushstrokes, visible canvas weave, "
            "rich layered pigment, chiaroscuro lighting, "
            "classical master technique"
        ),

        "sketch": (
            "GRAPHITE PENCIL DRAWING. Monochrome only. "
            "((High contrast charcoal sketch)), {subject}, "
            "cross-hatching, controlled shading, "
            "paper grain texture visible, "
            "no color, no paint, no digital effects"
        ),

        "cyberpunk": (
            "CYBERPUNK NEON NIGHT. "
            "((Neon noir futuristic scene)), {subject}, "
            "blue and magenta neon lighting, wet reflective streets, "
            "volumetric fog, holographic UI glow, "
            "high contrast night atmosphere"
        ),

        "fantasy art": (
            "EPIC FANTASY ILLUSTRATION. Medieval magical world. "
            "((High-end fantasy concept art)), {subject}, "
            "ornate armor, glowing particles, "
            "cinematic dramatic lighting, "
            "rich color depth, highly detailed environment"
        ),

        "omni": (
            "FOTOREALISTIC HIGH-RESOLUTION PHOTOGRAPH. "
            "((Ultra detailed, sharp focus, vivid colors)), {subject}, "
            "50mm lens f/1.8 shallow depth of field, subject sharp with natural background bokeh, "
            "fine skin texture, individual hair strands, fabric detail, natural soft lighting, "
            "balanced contrast, subtle shadows, professional color science, "
            "documentary quality, preserve original composition and structure"
        ),
    }

    # ===== 스타일별 Negative =====
    STYLE_NEG_DB: dict[str, str] = {

        "realistic": (
            "painting, drawing, cartoon, anime, CGI, "
            "3D render look, cel shading, stylized skin"
        ),

        "cinematic": (
            "cartoon, anime, flat lighting, amateur, phone camera, pixel art"
        ),

        "anime": (
            "3D render, photorealistic, photograph, western cartoon, "
            "real skin texture, volumetric lighting"
        ),

        "3d render": (
            "2D drawing, sketch, watercolor, oil painting, photograph"
        ),

        "pixel art": (
            "voxel, 3D blocks, Minecraft, lego, "
            "isometric perspective, smooth shading, gradient, "
            "realistic lighting, ray tracing"
        ),

        "watercolor": (
            "digital lines, vector art, CGI, oil paint, neon lighting"
        ),

        "oil painting": (
            "flat digital, anime, pixel art, sketch, photograph"
        ),

        "sketch": (
            "color, painted, digital paint, 3D render, photograph"
        ),

        "cyberpunk": (
            "daylight, nature, watercolor, oil painting, anime"
        ),

        "fantasy art": (
            "modern technology, sci-fi, pixel art, voxel, photograph"
        ),

        "omni": (
            "blurry face, wrong fingers, bad hands, extra fingers, "
            "text, watermark, logo, excessive noise, distortion, "
            "painting, cartoon, anime, lowres, oversaturated, plastic skin"
        ),
    }

    @classmethod
    def get_allowed_style_keys(cls) -> list[str]:
        """API 검증·스타일 목록용 단일 소스. 이 키만 허용하면 파이프라인과 항상 일치."""
        return sorted(cls.STYLE_POS_DB.keys())

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

        pos_template = cls.STYLE_POS_DB.get(style_key)
        neg_template = cls.STYLE_NEG_DB.get(style_key, "")

        user = (user_content or "").strip()

        if user:
            content = f"((({user})))"
        else:
            content = (
                "((keep original subject identity and subject count exactly, "
                "do not replace with human, do not merge subjects))"
            )

        final_positive = (
            pos_template.format(subject=content)
            + ", "
            + cls.BASE_QUALITY
            + f", aspect ratio {aspect_ratio}"
        )

        final_negative = cls.BASE_NEGATIVE + ", " + neg_template

        return {
            "style": style_key,
            "final_prompt": final_positive,
            "negative_prompt": final_negative,
            "consistency_hint": "Reuse same Seed and Gen ID to maintain identity consistency."
        }