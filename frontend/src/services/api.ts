import axios, { AxiosError } from "axios";
import type { GenerateResponse, ErrorDetail, StylesResponse } from "../types/api";

const api = axios.create({
  baseURL: "",
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

export function isApiError(error: unknown): error is AxiosError<ErrorDetail> {
  return axios.isAxiosError(error) && error.response?.data !== undefined;
}

export function getErrorMessage(error: unknown): string {
  if (isApiError(error)) {
    const detail = error.response?.data?.detail;
    if (typeof detail === "string") return detail;
    if (Array.isArray(detail)) {
      return detail.map((d) => (typeof d === "object" && d?.msg ? d.msg : String(d))).join(", ");
    }
  }
  if (error instanceof Error) return error.message;
  return "An unexpected error occurred";
}
