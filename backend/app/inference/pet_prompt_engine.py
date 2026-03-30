"""
강아지·고양이 I2V용 프롬프트 보강 (트리거 토큰 + 품질 suffix).

마스터 프롬프트 스펙의 ``PetPromptEngine`` 요지를 반영.
실제 영상 생성 파이프라인은 ``app.services.video_service`` 가 담당한다.
"""

from __future__ import annotations

import random
from typing import Literal

QualityCategory = Literal[
    "COOKING",
    "OFFICE",
    "TALKING",
    "HUMANIZE",
    "NATURAL",
    "SEASONAL",
    "FITNESS",
    "TRAVEL",
    "SLEEPY",
    "BIRTHDAY",
]


class PetPromptEngine:
    """트리거 토큰·품질 접미·네거티브·카테고리 템플릿."""

    QUALITY_SUFFIX = (
        "high quality video, smooth natural motion, photorealistic fur and eyes, "
        "sharp focus, good lighting, stable camera, no artifacts, natural pet anatomy"
    )

    DEFAULT_NEGATIVE = (
        "blurry, distorted, low quality, artifacts, watermark, text overlay, duplicate frames, "
        "jerky motion, overexposed, underexposed, deformed legs, extra limbs, human face swap, "
        "cartoon, anime, low resolution, flickering"
    )

    # id -> template with "{trigger}" placeholder (40개 이상)
    _TEMPLATES: dict[str, str] = {
        # COOKING
        "cook_1": "{trigger} wearing a white chef apron, stirring pasta in a large pot, professional kitchen, warm lighting, focused expression",
        "cook_2": "{trigger} chopping vegetables with paws on a wooden cutting board, cozy home kitchen, natural window light",
        "cook_3": "{trigger} decorating a birthday cake with frosting, bakery setting, colorful sprinkles, playful expression",
        "cook_4": "{trigger} holding a wooden spoon, tasting soup from a big pot, steam rising, warm golden lighting",
        "cook_5": "{trigger} kneading bread dough on a flour-dusted countertop, rustic kitchen, morning sunlight",
        "cook_6": "{trigger} flipping pancakes in a pan, small flame, cozy breakfast nook",
        "cook_7": "{trigger} arranging sushi on a plate, clean counter, soft overhead light",
        # OFFICE
        "off_1": "{trigger} wearing glasses, typing near a laptop, modern desk, city view, focused expression",
        "off_2": "{trigger} in a small suit presenting at a whiteboard, conference room, professional lighting",
        "off_3": "{trigger} coffee break holding a mug, office corner, bookshelf background",
        "off_4": "{trigger} on a video call looking at a monitor, home office, ring light",
        "off_5": "{trigger} signing papers with a pen, executive desk, formal tone",
        "off_6": "{trigger} rolling on an office chair, playful but tidy cubicle",
        # TALKING
        "talk_1": "{trigger} looking at camera, mouth slightly open, blinking naturally, neutral background, close-up",
        "talk_2": "{trigger} surprised expression, eyes wide, soft background blur",
        "talk_3": "{trigger} happy squint, relaxed ears, warm lighting",
        "talk_4": "{trigger} head tilt listening, subtle ear movement",
        # HUMANIZE
        "hum_1": "{trigger} walking confidently on a park path, sunny day, casual vibe",
        "hum_2": "{trigger} sitting at a cafe table, relaxed posture, afternoon light",
        "hum_3": "{trigger} holding a phone as if taking a selfie, city bokeh",
        "hum_4": "{trigger} reading a magazine on a sofa, cozy living room",
        # NATURAL
        "nat_1": "{trigger} running joyfully across a green meadow, golden hour, dynamic motion",
        "nat_2": "{trigger} playing with a colorful ball, sunny park, joyful energy",
        "nat_3": "{trigger} splashing in a shallow puddle, rainy day, water droplets",
        "nat_4": "{trigger} yawning and stretching on a soft bed, morning light",
        "nat_5": "{trigger} chasing falling autumn leaves, colorful trees, crisp air",
        "nat_6": "{trigger} digging gently in sand, beach breeze",
        # SEASONAL
        "sea_1": "{trigger} playing in fresh snow, winter wonderland, soft scarf",
        "sea_2": "{trigger} under cherry blossoms, spring breeze, pink petals",
        "sea_3": "{trigger} at the beach, waves in background, summer sunshine",
        "sea_4": "{trigger} jumping in autumn leaves, orange foliage",
        "sea_5": "{trigger} cozy by a fireplace, winter evening, warm tones",
        # FITNESS + TRAVEL + misc
        "fit_1": "{trigger} light jogging on a trail, healthy coat, morning jog",
        "fit_2": "{trigger} stretching after play, yoga mat, soft studio light",
        "trv_1": "{trigger} looking out a train window, scenic blur, curious eyes",
        "trv_2": "{trigger} sitting in a car back seat, road trip vibe, safety first",
        "slp_1": "{trigger} deep sleep curled in a donut bed, peaceful breathing",
        "slp_2": "{trigger} lazy blink on a windowsill, afternoon sun stripe",
        "bday_1": "{trigger} wearing a tiny party hat, confetti, soft bokeh",
        "bday_2": "{trigger} sniffing a cupcake with a single candle, gentle motion",
        "extra_1": "{trigger} catching treats mid-air, slow motion, bright kitchen",
        "extra_2": "{trigger} shaking off water after bath, clean tiles, droplets",
        "extra_3": "{trigger} meeting a butterfly in a garden, shallow depth of field",
        "extra_4": "{trigger} peeking from under a blanket, cute nose, warm lamp",
        "extra_5": "{trigger} riding in a pet stroller, urban park, smooth push",
    }

    _CATEGORY_IDS: dict[QualityCategory, list[str]] = {
        "COOKING": ["cook_1", "cook_2", "cook_3", "cook_4", "cook_5", "cook_6", "cook_7"],
        "OFFICE": ["off_1", "off_2", "off_3", "off_4", "off_5", "off_6"],
        "TALKING": ["talk_1", "talk_2", "talk_3", "talk_4"],
        "HUMANIZE": ["hum_1", "hum_2", "hum_3", "hum_4"],
        "NATURAL": ["nat_1", "nat_2", "nat_3", "nat_4", "nat_5", "nat_6"],
        "SEASONAL": ["sea_1", "sea_2", "sea_3", "sea_4", "sea_5"],
        "FITNESS": ["fit_1", "fit_2"],
        "TRAVEL": ["trv_1", "trv_2"],
        "SLEEPY": ["slp_1", "slp_2"],
        "BIRTHDAY": ["bday_1", "bday_2"],
    }

    @classmethod
    def list_categories(cls) -> list[str]:
        """템플릿 카테고리 키 목록."""
        return list(cls._CATEGORY_IDS.keys())

    @classmethod
    def list_template_ids(cls) -> list[str]:
        """등록된 템플릿 키 목록."""
        return sorted(cls._TEMPLATES.keys())

    @classmethod
    def render_template(cls, template_id: str, trigger_token: str) -> str:
        """단일 템플릿 문자열을 렌더링한다."""
        raw = cls._TEMPLATES.get(template_id)
        if not raw:
            raise KeyError(f"Unknown template_id: {template_id}")
        return raw.replace("{trigger}", trigger_token.strip())

    @classmethod
    def get_prompt(
        cls,
        category: QualityCategory,
        trigger_token: str,
        breed: str | None = None,
        template_id: str | None = None,
    ) -> tuple[str, str]:
        """
        (enhanced_prompt, negative_prompt) 반환.

        ``template_id`` 가 있으면 해당 템플릿만 사용. 없으면 카테고리에서 무작위 하나.
        """
        tid = template_id
        if not tid:
            pool = cls._CATEGORY_IDS.get(category) or list(cls._TEMPLATES.keys())
            tid = random.choice(pool)
        base = cls.render_template(tid, trigger_token)
        breed_hint = f", {breed} breed detail" if (breed or "").strip() else ""
        positive = f"{base}{breed_hint}, {cls.QUALITY_SUFFIX}"
        return positive, cls.DEFAULT_NEGATIVE

    @classmethod
    def enhance_custom_prompt(cls, raw_prompt: str, trigger_token: str) -> str:
        """사용자 입력 앞에 트리거와 품질 접미를 붙인다."""
        t = (raw_prompt or "").strip()
        tr = trigger_token.strip()
        if not t:
            return f"{tr}, {cls.QUALITY_SUFFIX}" if tr else cls.QUALITY_SUFFIX
        if tr and not t.lower().startswith(tr.lower()):
            return f"{tr}, {t}, {cls.QUALITY_SUFFIX}"
        return f"{t}, {cls.QUALITY_SUFFIX}"
