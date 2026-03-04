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

export type StyleKey =
  | "anime"
  | "realistic"
  | "watercolor"
  | "cyberpunk"
  | "oil painting"
  | "sketch"
  | "cinematic"
  | "fantasy art"
  | "pixel art"
  | "3d render";

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
