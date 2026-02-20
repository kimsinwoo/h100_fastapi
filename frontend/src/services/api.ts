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

// ---------- LLM (gpt-oss-20b) ----------

export async function getLlmStatus(): Promise<{ available: boolean; model: string | null }> {
  const { data } = await api.get<{ available: boolean; model: string | null }>("/api/llm/status");
  return data;
}

/** LLM 채팅 (건강 도우미). 첫 응답·재생성 시 100초 이상 걸릴 수 있으므로 타임아웃 4분. */
const LLM_CHAT_TIMEOUT_MS = 240_000;

export async function llmChat(
  messages: Array<{ role: string; content: string }>,
  maxTokens = 256,
  temperature = 0.7
): Promise<string> {
  const { data } = await api.post<{ content: string }>(
    "/api/llm/chat",
    { messages, max_tokens: maxTokens, temperature },
    { timeout: LLM_CHAT_TIMEOUT_MS }
  );
  return data.content;
}

export async function suggestPrompt(style: string, userHint?: string | null): Promise<string> {
  const { data } = await api.post<{ prompt: string }>("/api/llm/suggest-prompt", {
    style,
    user_hint: userHint || undefined,
  });
  return data.prompt;
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
  if (axios.isAxiosError(error)) {
    if (error.code === "ECONNABORTED" || (error.message && /timeout/i.test(error.message)))
      return "응답 시간이 초과되었습니다. 잠시 후 다시 시도해 주세요.";
    if (error.response?.data !== undefined) {
      const detail = error.response.data?.detail;
      if (typeof detail === "string") return detail;
      if (Array.isArray(detail)) {
        return (detail as Array<{ msg?: string } | string>)
          .map((d: { msg?: string } | string) =>
            typeof d === "object" && d && "msg" in d ? d.msg : String(d)
          )
          .join(", ");
      }
    }
  }
  if (error instanceof Error) return error.message;
  return "오류가 발생했습니다.";
}
