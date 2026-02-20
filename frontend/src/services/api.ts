import axios, { AxiosError } from "axios";
import type { GenerateResponse, ErrorDetail, StylesResponse, TrainingItem } from "../types/api";

const api = axios.create({
  baseURL: "http://210.91.154.131:20443/95ce287337c3ad9f",
  timeout: 120_000,
  headers: { "Content-Type": "application/json" },
});

export async function generateImage(
  file: File,
  style: string,
  customPrompt: string | null,
  strength: number | null,  
  seed: number | null
): Promise<GenerateResponse> {
  const form = new FormData();
  form.append("file", file);
  form.append("style", style);
  if (customPrompt !== null && customPrompt.trim() !== "") {
    form.append("custom_prompt", customPrompt.trim());
  }
  if (strength !== null) {
    form.append("strength", String(strength));
  }
  if (seed !== null) {
    form.append("seed", String(seed));
  }
  const { data } = await api.post<GenerateResponse>("/api/generate", form, {
    headers: { "Content-Type": "multipart/form-data" },
  });
  return data;
}

export async function getStyles(): Promise<StylesResponse> {
  const { data } = await api.get<StylesResponse>("/api/styles");
  return data;
}

export async function getHealth(): Promise<{ status: string; gpu_available: boolean }> {
  const { data } = await api.get<{ status: string; gpu_available: boolean }>("/health");
  return data;
}

// ---------- LoRA 학습 데이터 ----------

export async function getTrainingItems(): Promise<TrainingItem[]> {
  const { data } = await api.get<TrainingItem[]>("/api/training/items");
  return data;
}

export async function addTrainingItem(file: File, caption: string): Promise<TrainingItem> {
  const form = new FormData();
  form.append("file", file);
  form.append("caption", caption);
  const { data } = await api.post<TrainingItem>("/api/training/items", form, {
    headers: { "Content-Type": "multipart/form-data" },
  });
  return data;
}

export async function updateTrainingCaption(itemId: string, caption: string): Promise<TrainingItem> {
  const { data } = await api.patch<TrainingItem>(`/api/training/items/${itemId}`, { caption });
  return data;
}

export async function deleteTrainingItem(itemId: string): Promise<void> {
  await api.delete(`/api/training/items/${itemId}`);
}

export async function startTraining(): Promise<{ status: string; message?: string; error?: string }> {
  const { data } = await api.post<{ status: string; message?: string; error?: string }>("/api/training/start");
  return data;
}

/** 학습용 이미지 전체 URL (API base + image_url). */
export function getTrainingImageFullUrl(imageUrl: string): string {
  if (imageUrl.startsWith("http")) return imageUrl;
  const base = api.defaults.baseURL ?? "";
  return base ? `${base.replace(/\/$/, "")}${imageUrl.startsWith("/") ? "" : "/"}${imageUrl}` : imageUrl;
}

export function isApiError(error: unknown): error is AxiosError<ErrorDetail> {
  return axios.isAxiosError(error) && error.response?.data !== undefined;
}

export function getErrorMessage(error: unknown): string {
  if (isApiError(error)) {
    const detail = error.response?.data?.detail;
    if (typeof detail === "string") return detail;
    if (Array.isArray(detail)) {
      return (detail as Array<{ msg?: string } | string>)
        .map((d: { msg?: string } | string) =>
          typeof d === "object" && d && "msg" in d ? d.msg : String(d)
        )
        .join(", ");
    }
  }
  if (error instanceof Error) return error.message;
  return "An unexpected error occurred";
}
