export interface GenerateResponse {
  original_url: string;
  generated_url: string;
  processing_time: number;
  /** PNG base64; use when GET generated_url returns 404 (e.g. multi-pod) */
  generated_image_base64?: string | null;
}

export interface ErrorDetail {
  detail: string;
  code?: string;
}

/** 백엔드 허용 스타일 (prompt-config와 동기화). API /api/styles 응답 키와 일치 */
export type StyleKey =
  | "dragonball"
  | "slamdunk"
  | "sailor_moon"
  | "pokemon"
  | "dooly"
  | "mazinger"
  | "shinchan"
  | "pixel_art"
  | "pixel art"
  | "animal_crossing"
  | "ac_style_transfer"
  | "clay_art"
  | "cloud_theme";

export interface StylesResponse {
  [key: string]: string;
}

export interface TrainingItem {
  id: string;
  image_filename: string;
  image_url: string;
  caption: string;
  category?: string;
  created_at?: string;
}

// ---------- AC Villager Reconstruction ----------

/** Stage 1: biological analysis result (structure only) */
export interface ACBiologicalAnalysis {
  species: string;
  main_fur_color: string;
  secondary_fur_color: string;
  eye_color: string;
  markings: string;
  ear_type: string;
  tail_type: string;
}

/** Stage 2: request body (no image) */
export interface ACReconstructRequest {
  species: string;
  main_fur_color?: string;
  secondary_fur_color?: string;
  eye_color?: string;
  markings?: string;
  ear_type?: string | null;
  tail_type?: string | null;
  seed?: number | null;
}

// ---------- Image Analysis (structured visual attributes) ----------

export interface AnimalInfo {
  species: string;
  breed: string | null;
  fur_main_color: string;
  fur_secondary_color: string;
  major_markings: string;
}

export interface ClothingDetection {
  is_wearing_clothes: boolean;
  clothing_type: string;
  clothing_color: string;
  clothing_pattern: string;
  sleeve_length: string;
  full_body_outfit: boolean;
}

export interface Accessories {
  hat: string;
  glasses: string;
  collar: string;
  ribbon: string;
  other_visible_accessory: string;
}

export interface Pose {
  posture: string;
  facing_direction: string;
  tail_position: string;
}

export interface Environment {
  setting: string;
  dominant_background_colors: string;
}

export interface ImageAnalysisResponse {
  animal: AnimalInfo;
  clothing: ClothingDetection;
  accessories: Accessories;
  pose: Pose;
  environment: Environment;
}

/** Viewpoint / camera angle analysis. JSON only. */
export interface ViewpointAnalysisResponse {
  view_angle: "front" | "three-quarter" | "side-profile-left" | "side-profile-right";
  head_visible_eyes: 1 | 2;
  body_orientation_degrees: number;
  tail_visible: boolean;
}

/** Universal analysis: pose, camera, gravity, clothing, structure. JSON only. */
export interface UniversalAnalysisResponse {
  species: string;
  view_angle: string;
  body_pose: string;
  gravity_axis: string;
  head_direction_degrees: number;
  spine_alignment: string;
  visible_eyes: number;
  leg_visibility_count: number;
  is_full_body_visible: boolean;
  is_wearing_clothes: boolean;
  clothing_type: string;
  clothing_color: string;
  clothing_pattern: string;
  clothing_confidence: number;
}
