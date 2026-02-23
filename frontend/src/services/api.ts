import axios, { AxiosError } from "axios";
import type { GenerateResponse, ErrorDetail, StylesResponse, TrainingItem } from "../types/api";

const api = axios.create({
  baseURL: "http://210.91.154.131:20443/vscode/h8212918284d84e9b348b302527193731-3228-0/proxy/7000",
  timeout: 120_000,
  headers: { "Content-Type": "application/json" },
});

const USER_ID_KEY = "user_id";

/** 로그인 없이 브라우저 단위 격리용. 최초 접속 시 UUID 생성 후 localStorage에 저장. */
export function getOrCreateUserId(): string {
  let id = localStorage.getItem(USER_ID_KEY);
  if (!id || !id.trim()) {
    id = crypto.randomUUID?.() ?? "xxxxxxxx-xxxx-4xxx-yxxx-xxxxxxxxxxxx".replace(/[xy]/g, (c) => {
      const r = (Math.random() * 16) | 0;
      const v = c === "x" ? r : (r & 0x3) | 0x8;
      return v.toString(16);
    });
    localStorage.setItem(USER_ID_KEY, id);
  }
  return id;
}

api.interceptors.request.use((config) => {
  if (config.url?.includes("/api/chat")) {
    config.headers.set("X-User-Id", getOrCreateUserId());
  }
  return config;
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

// ---------- 채팅방 저장 ----------

export type ChatRoomSummary = { id: string; title: string; updated_at: string };
export type ChatRoom = { id: string; title: string; messages: Array<{ role: string; content: string }>; created_at: string; updated_at: string };

export async function getChatRooms(): Promise<ChatRoomSummary[]> {
  const { data } = await api.get<ChatRoomSummary[]>("/api/chat/rooms", {
    headers: { "X-User-Id": getOrCreateUserId() },
  });
  return data;
}

export async function getChatRoom(roomId: string): Promise<ChatRoom> {
  const { data } = await api.get<ChatRoom>(`/api/chat/rooms/${roomId}`, {
    headers: { "X-User-Id": getOrCreateUserId() },
  });
  return data;
}

export async function createChatRoom(title?: string): Promise<ChatRoom> {
  const { data } = await api.post<ChatRoom>(
    "/api/chat/rooms",
    { title: title || undefined },
    { headers: { "X-User-Id": getOrCreateUserId() } }
  );
  return data;
}

export async function addChatMessage(roomId: string, role: "user" | "assistant", content: string): Promise<ChatRoom> {
  const { data } = await api.post<ChatRoom>(
    `/api/chat/rooms/${roomId}/messages`,
    { role, content },
    { headers: { "X-User-Id": getOrCreateUserId() } }
  );
  return data;
}

export async function deleteChatRoom(roomId: string): Promise<void> {
  await api.delete(`/api/chat/rooms/${roomId}`, {
    headers: { "X-User-Id": getOrCreateUserId() },
  });
}

// ---------- LLM ----------

/** LLM 채팅 (건강 도우미). 응답에 5분 이상 걸릴 수 있으므로 타임아웃 10분. */
const LLM_CHAT_TIMEOUT_MS = 600_000;

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
    if (error.code === "ERR_NETWORK" || (error.message && /network error/i.test(error.message)))
      return "연결이 끊겼습니다. LLM 응답에 1~2분 이상 걸릴 수 있어 중간에 끊겼을 수 있습니다. 다시 시도해 주세요.";
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
