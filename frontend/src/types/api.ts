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
  | "clay_art";

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
