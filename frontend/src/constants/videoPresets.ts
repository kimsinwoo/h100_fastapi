/**
 * 이미지 기반 영상 생성용 프롬프트 — 주인 시점(POV), 강아지만 등장, 원본 유지.
 * 카메라=주인 / 사람 등장 금지 / 손만 잠깐 허용.
 * 안정성 핵심: first person perspective, only the dog visible, owner's hand may briefly appear.
 */

const COMMON_NEGATIVE =
  "person visible, human body, human face, full human, extra people, multiple dogs, different dog, changing breed, distorted body, mutated limbs, unrealistic motion, glitch, blur, changing background";

export type VideoPresetItem = {
  id: string;
  label: string;
  prompt: string;
  negative: string;
};

/** 기본 8종 — 성공률 높은 순으로 배치. 1~3이 가장 안정적. */
export const VIDEO_PRESETS: VideoPresetItem[] = [
  {
    id: "toy_shake",
    label: "주인이 장난감을 흔들어주는 상황 (가장 안정적)",
    prompt:
      "first person perspective from the dog's owner holding the camera, only the dog is visible in the scene, the exact same dog from the original image excitedly jumping and trying to catch a toy being shaken in front of the camera, the owner's hand briefly appears in front of the camera holding the toy, playful barking, wagging tail rapidly, energetic happy movement, the dog focuses on the toy near the camera, natural body motion, realistic physics, original background unchanged, smooth animation, high detail fur",
    negative: COMMON_NEGATIVE,
  },
  {
    id: "greet_camera",
    label: "카메라 가까이 와서 반겨주는 강아지",
    prompt:
      "first person perspective from the dog's owner holding the camera, only the dog is visible, the exact same dog from the original image runs happily toward the camera and greets the owner excitedly, wagging tail rapidly, small jumps toward the camera, joyful barking, playful and affectionate behavior, the dog looks directly into the camera, natural body movement, realistic physics, original background unchanged, smooth animation",
    negative: COMMON_NEGATIVE,
  },
  {
    id: "dance_front",
    label: "카메라 앞에서 춤추는 강아지",
    prompt:
      "first person perspective from the dog's owner holding the camera, only the dog appears in the scene, the exact same dog from the original image happily dancing in front of the camera, rhythmic playful movements, spinning slightly and jumping with excitement, wagging tail, joyful barking, the dog performs cute energetic dance moves while looking toward the camera, natural body motion, realistic physics, original background unchanged, smooth animation",
    negative: COMMON_NEGATIVE,
  },
  {
    id: "ball_wait",
    label: "카메라 앞에서 공 기다리는 강아지",
    prompt:
      "first person perspective from the dog's owner holding the camera, only the dog visible, the exact same dog from the original image sits excitedly in front of the camera waiting for a ball, the owner's hand briefly appears holding the ball near the camera, the dog watches the ball intensely, wagging tail rapidly, small excited barks, energetic anticipation, natural body posture, realistic physics, original background unchanged, smooth animation",
    negative: COMMON_NEGATIVE,
  },
  {
    id: "treat_wait",
    label: "카메라 앞에서 간식 기다리는 강아지",
    prompt:
      "first person perspective from the dog's owner holding the camera, only the dog appears in the scene, the exact same dog from the original image sits attentively in front of the camera, the owner's hand briefly appears holding a treat close to the camera, the dog focuses intensely on the treat, wagging tail slowly with excitement, small happy barks, bright curious eyes, natural body movement, realistic physics, original background unchanged",
    negative: COMMON_NEGATIVE,
  },
  {
    id: "run_around_camera",
    label: "카메라 주위를 뛰어다니는 강아지",
    prompt:
      "first person perspective from the dog's owner holding the camera, only the dog visible in the scene, the exact same dog from the original image runs playfully around the camera with energetic movement, wagging tail rapidly, joyful barking, occasionally running close to the camera, lively playful personality, natural body motion, realistic physics, original background unchanged, smooth animation",
    negative: COMMON_NEGATIVE,
  },
  {
    id: "toy_in_mouth",
    label: "카메라 앞에서 장난감 물고 흔드는 강아지",
    prompt:
      "first person perspective from the dog's owner holding the camera, only the dog appears in the scene, the exact same dog from the original image grabs a toy and shakes it playfully in front of the camera, wagging tail rapidly, playful barking, energetic happy movement, the dog proudly shows the toy near the camera, natural body motion, realistic physics, original background unchanged, smooth animation",
    negative: COMMON_NEGATIVE,
  },
  {
    id: "lie_down",
    label: "카메라 앞에서 편하게 눕는 강아지",
    prompt:
      "first person perspective from the dog's owner holding the camera, only the dog appears in the scene, the exact same dog from the original image slowly lies down near the camera and relaxes comfortably, calm breathing, gentle tail movement, peaceful moment, the dog looks toward the camera affectionately, natural body motion, realistic physics, original background unchanged, soft cinematic lighting",
    negative: COMMON_NEGATIVE,
  },
];

/** AI 영상 생성에서 성공률 높은 강아지 POV 프롬프트 TOP 20 */
export const VIDEO_PRESETS_TOP_20: VideoPresetItem[] = [
  {
    id: "t20_toy_shake",
    label: "장난감 흔들기 (손 등장)",
    prompt:
      "first person perspective from the dog's owner holding the camera, only the dog is visible in the scene, the exact same dog from the original image excitedly jumping and trying to catch a toy being shaken in front of the camera, the owner's hand briefly appears in front of the camera holding the toy, playful barking, wagging tail rapidly, energetic happy movement, natural body motion, realistic physics, original background unchanged, smooth animation, high detail fur",
    negative: COMMON_NEGATIVE,
  },
  {
    id: "t20_greet",
    label: "카메라 향해 뛰어와 반기",
    prompt:
      "first person perspective from the dog's owner holding the camera, only the dog is visible, the exact same dog from the original image runs happily toward the camera and greets the owner excitedly, wagging tail rapidly, small jumps toward the camera, joyful barking, the dog looks directly into the camera, natural body movement, realistic physics, original background unchanged, smooth animation",
    negative: COMMON_NEGATIVE,
  },
  {
    id: "t20_treat_hand",
    label: "손에 든 간식 주시",
    prompt:
      "first person perspective from the dog's owner holding the camera, only the dog appears in the scene, the exact same dog from the original image sits attentively in front of the camera, the owner's hand briefly appears holding a treat close to the camera, the dog focuses intensely on the treat, wagging tail slowly, bright curious eyes, natural body movement, realistic physics, original background unchanged",
    negative: COMMON_NEGATIVE,
  },
  {
    id: "t20_dance",
    label: "카메라 앞에서 춤",
    prompt:
      "first person perspective from the dog's owner holding the camera, only the dog appears in the scene, the exact same dog from the original image happily dancing in front of the camera, rhythmic playful movements, spinning and jumping with excitement, wagging tail, joyful barking, the dog looks toward the camera, natural body motion, realistic physics, original background unchanged, smooth animation",
    negative: COMMON_NEGATIVE,
  },
  {
    id: "t20_head_tilt",
    label: "고개 갸우뚱 호기심",
    prompt:
      "first person perspective from the dog's owner holding the camera, only the dog is visible in the scene, the exact same dog from the original image tilts its head curiously toward the camera, ears perked, bright attentive eyes, slight tail wag, cute confused expression, natural body motion, realistic physics, original background unchanged, smooth animation",
    negative: COMMON_NEGATIVE,
  },
  {
    id: "t20_spin",
    label: "신나서 빙글빙글",
    prompt:
      "first person perspective from the dog's owner holding the camera, only the dog visible in the scene, the exact same dog from the original image spins in place with excitement in front of the camera, wagging tail wildly, joyful barking, energetic happy movement, the dog looks toward the camera, natural body motion, realistic physics, original background unchanged, smooth animation",
    negative: COMMON_NEGATIVE,
  },
  {
    id: "t20_paw_camera",
    label: "카메라 향해 발 터치",
    prompt:
      "first person perspective from the dog's owner holding the camera, only the dog appears in the scene, the exact same dog from the original image playfully paws toward the camera, wagging tail, excited expression, wanting attention, natural body motion, realistic physics, original background unchanged, smooth animation",
    negative: COMMON_NEGATIVE,
  },
  {
    id: "t20_sniff",
    label: "카메라 쪽 코로 킁킁",
    prompt:
      "first person perspective from the dog's owner holding the camera, only the dog is visible in the scene, the exact same dog from the original image sniffs curiously toward the camera, nose twitching, attentive expression, natural body posture, realistic physics, original background unchanged, smooth animation",
    negative: COMMON_NEGATIVE,
  },
  {
    id: "t20_sit_wag",
    label: "앉아서 꼬리 흔들며 대기",
    prompt:
      "first person perspective from the dog's owner holding the camera, only the dog visible in the scene, the exact same dog from the original image sits in front of the camera wagging its tail happily, looking at the camera with anticipation, natural body posture, realistic physics, original background unchanged, smooth animation",
    negative: COMMON_NEGATIVE,
  },
  {
    id: "t20_play_bow",
    label: "플레이 바우 (앞다리 숙이고 설레는 자세)",
    prompt:
      "first person perspective from the dog's owner holding the camera, only the dog appears in the scene, the exact same dog from the original image does a playful bow in front of the camera, front legs down, rear up, wagging tail, excited to play, natural body motion, realistic physics, original background unchanged, smooth animation",
    negative: COMMON_NEGATIVE,
  },
  {
    id: "t20_roll",
    label: "바닥에 뒹굴뒹굴",
    prompt:
      "first person perspective from the dog's owner holding the camera, only the dog is visible in the scene, the exact same dog from the original image rolls on the ground playfully in front of the camera, wagging tail, happy relaxed movement, natural body motion, realistic physics, original background unchanged, smooth animation",
    negative: COMMON_NEGATIVE,
  },
  {
    id: "t20_yawn_stretch",
    label: "하품하고 스트레칭",
    prompt:
      "first person perspective from the dog's owner holding the camera, only the dog appears in the scene, the exact same dog from the original image yawns and stretches comfortably in front of the camera, relaxed posture, gentle movement, natural body motion, realistic physics, original background unchanged, soft lighting",
    negative: COMMON_NEGATIVE,
  },
  {
    id: "t20_bark_excited",
    label: "카메라 보며 짖기",
    prompt:
      "first person perspective from the dog's owner holding the camera, only the dog visible in the scene, the exact same dog from the original image barks excitedly toward the camera, wagging tail, energetic expression, natural body motion, realistic physics, original background unchanged, smooth animation",
    negative: COMMON_NEGATIVE,
  },
  {
    id: "t20_walk_toward",
    label: "천천히 카메라 쪽 걸어오기",
    prompt:
      "first person perspective from the dog's owner holding the camera, only the dog is visible in the scene, the exact same dog from the original image walks slowly toward the camera, calm curious steps, tail swaying gently, looking at the camera, natural body movement, realistic physics, original background unchanged, smooth animation",
    negative: COMMON_NEGATIVE,
  },
  {
    id: "t20_catch_toy",
    label: "공중에서 장난감 잡기",
    prompt:
      "first person perspective from the dog's owner holding the camera, only the dog appears in the scene, the exact same dog from the original image jumps to catch a toy in mid-air in front of the camera, the owner's hand briefly visible throwing the toy, energetic leap, wagging tail, natural body motion, realistic physics, original background unchanged, smooth animation",
    negative: COMMON_NEGATIVE,
  },
  {
    id: "t20_rest_paws",
    label: "앞발 모으고 카메라 바라보기",
    prompt:
      "first person perspective from the dog's owner holding the camera, only the dog visible in the scene, the exact same dog from the original image rests with front paws together looking affectionately at the camera, calm peaceful expression, gentle tail movement, natural body posture, realistic physics, original background unchanged, soft cinematic lighting",
    negative: COMMON_NEGATIVE,
  },
  {
    id: "t20_ball_wait",
    label: "공 들고 기다리기 (손 등장)",
    prompt:
      "first person perspective from the dog's owner holding the camera, only the dog visible, the exact same dog from the original image sits excitedly in front of the camera waiting for a ball, the owner's hand briefly appears holding the ball near the camera, the dog watches the ball intensely, wagging tail rapidly, natural body posture, realistic physics, original background unchanged, smooth animation",
    negative: COMMON_NEGATIVE,
  },
  {
    id: "t20_run_around",
    label: "카메라 주변 뛰어다니기",
    prompt:
      "first person perspective from the dog's owner holding the camera, only the dog visible in the scene, the exact same dog from the original image runs playfully around the camera, wagging tail rapidly, joyful barking, occasionally running close to the camera, natural body motion, realistic physics, original background unchanged, smooth animation",
    negative: COMMON_NEGATIVE,
  },
  {
    id: "t20_toy_proud",
    label: "장난감 물고 자랑하기",
    prompt:
      "first person perspective from the dog's owner holding the camera, only the dog appears in the scene, the exact same dog from the original image holds a toy in its mouth and shakes it playfully in front of the camera, wagging tail rapidly, proud happy expression, natural body motion, realistic physics, original background unchanged, smooth animation",
    negative: COMMON_NEGATIVE,
  },
  {
    id: "t20_lie_relax",
    label: "카메라 앞에서 편하게 눕기",
    prompt:
      "first person perspective from the dog's owner holding the camera, only the dog appears in the scene, the exact same dog from the original image slowly lies down near the camera and relaxes comfortably, calm breathing, gentle tail movement, the dog looks toward the camera affectionately, natural body motion, realistic physics, original background unchanged, soft cinematic lighting",
    negative: COMMON_NEGATIVE,
  },
];

/** 바이럴 쇼츠용 강아지 POV 컨셉 — 짧은 영상·리els용 */
export const VIDEO_PRESETS_VIRAL: VideoPresetItem[] = [
  {
    id: "v_pov_came_home",
    label: "POV: 집에 왔을 때 강아지 반응",
    prompt:
      "first person perspective from the dog's owner holding the camera just opened the door, only the dog is visible in the scene, the exact same dog from the original image runs toward the camera with extreme excitement, wagging tail rapidly, jumping happily, joyful barking, the dog looks directly into the camera, as if owner just came home, natural body motion, realistic physics, original background unchanged, smooth animation, viral moment",
    negative: COMMON_NEGATIVE,
  },
  {
    id: "v_pov_about_to_throw",
    label: "POV: 공 던지기 직전 강아지",
    prompt:
      "first person perspective from the dog's owner holding the camera and a ball, only the dog visible in the scene, the exact same dog from the original image sits in front of the camera staring intensely at the ball, the owner's hand briefly visible holding the ball, wagging tail rapidly, ready to run, explosive anticipation, natural body posture, realistic physics, original background unchanged, smooth animation, viral moment",
    negative: COMMON_NEGATIVE,
  },
  {
    id: "v_pov_treat_or_trick",
    label: "POV: 간식 보여주는 순간",
    prompt:
      "first person perspective from the dog's owner holding the camera and a treat, only the dog appears in the scene, the exact same dog from the original image freezes and stares at the treat with laser focus, the owner's hand briefly visible, wagging tail slowly, droopy cute eyes, maximum attention, natural body motion, realistic physics, original background unchanged, smooth animation, viral moment",
    negative: COMMON_NEGATIVE,
  },
  {
    id: "v_pov_morning_dog",
    label: "POV: 아침에 깨우는 강아지",
    prompt:
      "first person perspective from the dog's owner holding the camera in bed, only the dog is visible in the scene, the exact same dog from the original image approaches the camera excitedly with morning energy, wagging tail, gentle nudging toward camera, good morning vibe, natural body motion, realistic physics, original background unchanged, soft morning lighting, viral moment",
    negative: COMMON_NEGATIVE,
  },
  {
    id: "v_pov_walk_start",
    label: "POV: 산책 시작할 때",
    prompt:
      "first person perspective from the dog's owner holding the camera at the door, only the dog visible in the scene, the exact same dog from the original image runs in circles in front of the camera with excitement, wagging tail wildly, can't wait to go out, natural body motion, realistic physics, original background unchanged, smooth animation, viral moment",
    negative: COMMON_NEGATIVE,
  },
  {
    id: "v_pov_zoomies",
    label: "POV: 강아지 줌비스 (미친 댄스)",
    prompt:
      "first person perspective from the dog's owner holding the camera, only the dog appears in the scene, the exact same dog from the original image runs in random energetic bursts in front of the camera, zoomies, spinning and sprinting playfully, wagging tail rapidly, chaotic happy energy, natural body motion, realistic physics, original background unchanged, smooth animation, viral moment",
    negative: COMMON_NEGATIVE,
  },
  {
    id: "v_pov_guilty",
    label: "POV: 잘못한 뒤 강아지 표정",
    prompt:
      "first person perspective from the dog's owner holding the camera, only the dog is visible in the scene, the exact same dog from the original image looks at the camera with guilty cute expression, ears slightly back, slow tail wag, avoiding eye contact then peeking at camera, natural body posture, realistic physics, original background unchanged, smooth animation, viral moment",
    negative: COMMON_NEGATIVE,
  },
  {
    id: "v_pov_toy_war",
    label: "POV: 장난감 빼앗기 놀이",
    prompt:
      "first person perspective from the dog's owner holding the camera and a toy, only the dog visible in the scene, the exact same dog from the original image tugs playfully at the toy near the camera, the owner's hand briefly visible, growling playfully, wagging tail, tug of war, natural body motion, realistic physics, original background unchanged, smooth animation, viral moment",
    negative: COMMON_NEGATIVE,
  },
];

/** 강아지 영상을 더 자연스럽게 만드는 프롬프트 구조 — 앞부분에 넣을 핵심 구문 */
export const NATURAL_PROMPT_PREFIX = [
  "first person perspective from the dog's owner holding the camera",
  "only the dog visible in the scene",
  "the owner's hand may briefly appear near the camera",
];

export const NATURAL_PROMPT_TIPS = [
  "항상 'the exact same dog from the original image'로 동일 강아지 유지.",
  "행동은 한 가지에 집중: 한 문장으로 '무엇을 하는지' 명확히.",
  "'natural body motion, realistic physics'로 물리감 추가.",
  "'original background unchanged'로 배경 고정.",
  "손 등장 시 'the owner's hand briefly appears'로 짧게만 노출.",
];
