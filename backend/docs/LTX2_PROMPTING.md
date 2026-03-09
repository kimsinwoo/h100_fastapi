# LTX-2 Cinematic Prompting Guide

Lightricks LTX-2 / LTX-2-TURBO용 시네마틱 프롬프트 설계 규칙. 짧은 아이디어(idea)를 **6하 원칙(6W)** + 공식 시네마틱 구조에 맞춰 변환할 때 사용.

---

## Goal

- Scene coherence  
- Motion stability  
- Cinematic realism  
- Prompt adherence  

---

## Mandatory Structure: 6W (6하 원칙)

| Element | Include |
|--------|--------|
| **WHO** | Main subject, appearance, clothing/visual details |
| **WHERE** | Environment, location, atmosphere, weather/lighting |
| **WHEN** | Time of day, lighting, temporal pacing |
| **WHAT** | Action sequence, subject behavior, motion progression |
| **WHY** | Visual storytelling, motivation/emotion via physical cues |
| **HOW** | Camera shot type, movement, style, lens/composition |

---

## LTX-2 Cinematic Rules

1. One continuous paragraph  
2. 4–8 descriptive sentences  
3. Present-tense verbs  
4. Explicit camera movement description  
5. Lighting, atmosphere, textures  
6. Motion progression  
7. Clear temporal flow  

---

## Camera Language (Required)

Use: **wide shot**, **close-up**, **medium shot**, **tracking shot**, **dolly in/out**, **handheld camera**, **over-the-shoulder shot**, **slow pan**, **static shot**, **locked camera**.

---

## Visual Detailing

- Lighting: soft morning light, neon reflections, warm golden sunlight  
- Texture/atmosphere: cinematic film grain, fog drifting, wet pavement  
- Color and particles: color palette, atmospheric particles  

---

## Action Structure

1. Opening shot  
2. Motion progression  
3. Camera movement  
4. Final framing  

---

## Negative Constraints (Avoid)

- Readable text, logos/signage  
- Chaotic physics, too many characters  
- Contradictory lighting  

---

## Output Format for "Idea → Prompt"

**SECTION 1**  
Optimized LTX-2 prompt (one paragraph, 4–8 sentences, present tense).

**SECTION 2**  
Explanation: how the prompt satisfies WHO / WHERE / WHEN / WHAT / WHY / HOW.

---

## Example

**Idea:** Person turns head on a rainy night street.

**Prompt:**  
A cinematic medium shot of a young woman standing on a rain-soaked street at night. Neon lights reflect across the wet pavement as light fog drifts through the air. She slowly turns her head toward the camera, her eyes catching the glow of passing headlights. The camera gently pushes forward in a slow dolly movement, settling into a close-up as raindrops shimmer on her coat. Distant traffic hums softly in the background.

**Explanation:**  
WHO: young woman. WHERE: rain-soaked street. WHEN: night. WHAT: turning toward camera. WHY: subtle emotional tension. HOW: medium shot → dolly-in close-up.

---

## Integration

- **VIDEO_PROMPT_PRESETS** in `app.api.routes`: 각 프리셋은 위 구조를 따르도록 작성.  
- **idea → prompt** API/LLM 호출 시: 이 문서를 system/instruction으로 주고, 사용자 `idea`를 SECTION 1+2 형식으로 변환하도록 유도.
