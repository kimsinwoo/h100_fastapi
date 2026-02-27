import axios, { AxiosError } from "axios";
import type { GenerateResponse, ErrorDetail, StylesResponse, TrainingItem } from "../types/api";

/** 프론트 요청 타임아웃: 2분 (이미지 생성 등 긴 API용) */
const FRONTEND_TIMEOUT_MS = 2 * 60 * 1000; // 120_000

// 로컬 개발: .env에 VITE_API_BASE_URL=http://localhost:7000 설정 (프론트 3000, 백엔드 7000)
// 프로덕션: 같은 origin이면 비워두고, 프록시 쓰면 해당 URL 설정
const api = axios.create({
  baseURL: (import.meta as { env?: { VITE_API_BASE_URL?: string } }).env?.VITE_API_BASE_URL ?? "http://210.91.154.131:20443/95ce287337c3ad9f",
  timeout: FRONTEND_TIMEOUT_MS,
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
  // FormData 전송 시 기본 application/json 제거 → axios가 multipart/form-data; boundary=... 설정
  if (config.data instanceof FormData && config.headers) {
    delete (config.headers as Record<string, unknown>)["Content-Type"];
  }
  return config;
});

/** FormData 전용: Content-Type 미설정 → multipart/form-data; boundary= 자동 설정 (422 방지) */
const uploadApi = axios.create({
  baseURL: api.defaults.baseURL ?? "/",
  timeout: FRONTEND_TIMEOUT_MS,
});

/** Z-Image / SDXL: style + image. Z-Image 응답은 original_url, generated_url, processing_time */
export async function generateImage(
  file: File,
  style: string,
  customPrompt: string | null,
  strength: number | null,
  seed: number | null
): Promise<GenerateResponse> {
  const form = new FormData();
  form.append("style", style);
  form.append("image", file);
  form.append("custom_prompt", (customPrompt ?? "").trim());
  form.append("strength", String(strength ?? 0.75));
  form.append("steps", "20");
  form.append("cfg", "7.5");
  if (seed !== null) form.append("seed", String(seed));
  form.append("width", "1024");
  form.append("height", "1024");

  type ApiResponse = {
    original_url?: string;
    generated_url?: string;
    image_url?: string;
    processing_time?: number;
    processing_time_seconds?: number;
  };
  const { data } = await uploadApi.post<ApiResponse>("/api/generate", form);
  return {
    original_url: data.original_url ?? "",
    generated_url: data.generated_url ?? data.image_url ?? "",
    processing_time: data.processing_time ?? data.processing_time_seconds ?? 0,
  };
}

export async function getStyles(): Promise<StylesResponse> {
  const { data } = await api.get<StylesResponse>("/api/styles");
  return data;
}

/** 백엔드가 SDXL( main_sdxl )이면 /api/health, Z-Image( main )이면 /health — SDXL 기준으로 통일 */
export async function getHealth(): Promise<{ status: string; gpu_available: boolean }> {
  const { data } = await api.get<{ status: string; gpu_available: boolean }>("/api/health");
  return data;
}

// ---------- LLM (gpt-oss-20b) ----------

/** SDXL 백엔드에는 없음 → 404 시 사용 불가로 처리 */
export async function getLlmStatus(): Promise<{ available: boolean; model: string | null }> {
  try {
    const { data } = await api.get<{ available: boolean; model: string | null }>("/api/llm/status");
    return data;
  } catch {
    return { available: false, model: null };
  }
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

/** 건강 상담 구조화 응답 (감별 1~4순위, 응급 기준, 핵심 질문, 추천 카테고리) */
export type HealthChatDifferentialItem = {
  rank: number;
  name: string;
  reason: string;
  emergency: boolean;
  home_check: string;
};
export type HealthChatRecommendedCategory = { label: string; query: string };
export type HealthChatStructured = {
  differential: HealthChatDifferentialItem[];
  emergency_criteria: string[];
  key_questions: string[];
  recommended_categories: HealthChatRecommendedCategory[];
};

export type LlmChatResponse = { content: string; structured?: HealthChatStructured };

const getBaseUrl = () =>
  (import.meta as { env?: { VITE_API_BASE_URL?: string } }).env?.VITE_API_BASE_URL ??
  "http://210.91.154.131:20443/95ce287337c3ad9f";

/** 스트리밍 채팅: 청크마다 onChunk 호출, 완료 시 전체 content 반환. 표시가 빨리 되고 체감 속도 개선. */
export async function llmChatStream(
  messages: Array<{ role: string; content: string }>,
  onChunk: (chunk: string) => void,
  maxTokens = 1024,
  temperature = 0.4
): Promise<string> {
  const base = getBaseUrl();
  const url = base.replace(/\/$/, "") + "/api/llm/chat/stream";
  const res = await fetch(url, {
    method: "POST",
    headers: {
      "Content-Type": "application/json",
      "X-User-Id": getOrCreateUserId(),
    },
    body: JSON.stringify({
      messages,
      max_tokens: maxTokens,
      temperature,
    }),
    signal: AbortSignal.timeout(LLM_CHAT_TIMEOUT_MS),
  });
  if (!res.ok) {
    const errBody = await res.text();
    let msg = `HTTP ${res.status}`;
    try {
      const d = JSON.parse(errBody);
      if (d.detail) msg = typeof d.detail === "string" ? d.detail : JSON.stringify(d.detail);
    } catch {
      if (errBody) msg = errBody.slice(0, 200);
    }
    throw new Error(msg);
  }
  const reader = res.body?.getReader();
  if (!reader) throw new Error("No response body");
  const dec = new TextDecoder();
  let full = "";
  let buf = "";
  for (;;) {
    const { done, value } = await reader.read();
    if (done) break;
    buf += dec.decode(value, { stream: true });
    const lines = buf.split("\n");
    buf = lines.pop() ?? "";
    for (const line of lines) {
      const t = line.trim();
      if (!t) continue;
      try {
        const o = JSON.parse(t) as { content?: string };
        if (typeof o.content === "string" && o.content) {
          full += o.content;
          onChunk(o.content);
        }
      } catch {
        // ignore malformed line
      }
    }
  }
  if (buf.trim()) {
    try {
      const o = JSON.parse(buf.trim()) as { content?: string };
      if (typeof o.content === "string" && o.content) {
        full += o.content;
        onChunk(o.content);
      }
    } catch {
      // ignore
    }
  }
  return full;
}

export async function llmChat(
  messages: Array<{ role: string; content: string }>,
  maxTokens = 1024,
  temperature = 0.4
): Promise<LlmChatResponse> {
  const { data } = await api.post<LlmChatResponse>(
    "/api/llm/chat",
    { messages, max_tokens: maxTokens, temperature },
    { timeout: LLM_CHAT_TIMEOUT_MS }
  );
  return { content: data.content ?? "", structured: data.structured };
}

export async function suggestPrompt(style: string, userHint?: string | null): Promise<string> {
  const { data } = await api.post<{ prompt: string }>("/api/llm/suggest-prompt", {
    style,
    user_hint: userHint || undefined,
  });
  return data.prompt;
}

// ---------- LoRA 학습 데이터 ----------

export async function getTrainingItems(category?: string | null): Promise<TrainingItem[]> {
  const params = category ? { category } : {};
  const { data } = await api.get<TrainingItem[]>("/api/training/items", { params });
  return data;
}

export async function getTrainingCategories(): Promise<string[]> {
  const { data } = await api.get<string[]>("/api/training/categories");
  return data;
}

export async function addTrainingItem(file: File, caption: string, category?: string): Promise<TrainingItem> {
  const form = new FormData();
  form.append("file", file);
  form.append("caption", caption);
  form.append("category", category ?? "");
  const { data } = await api.post<TrainingItem>("/api/training/items", form, {
    headers: { "Content-Type": "multipart/form-data" },
  });
  return data;
}

export async function updateTrainingCaption(itemId: string, caption: string): Promise<TrainingItem> {
  const { data } = await api.patch<TrainingItem>(`/api/training/items/${itemId}`, { caption });
  return data;
}

export async function updateTrainingItem(
  itemId: string,
  updates: { caption?: string; category?: string }
): Promise<TrainingItem> {
  const { data } = await api.patch<TrainingItem>(`/api/training/items/${itemId}`, updates);
  return data;
}

export async function deleteTrainingItem(itemId: string): Promise<void> {
  await api.delete(`/api/training/items/${itemId}`);
}

export async function startTraining(category?: string | null): Promise<{
  status: string;
  message?: string;
  error?: string;
}> {
  const body = category ? { category } : {};
  const { data } = await api.post<{ status: string; message?: string; error?: string }>(
    "/api/training/start",
    body
  );
  return data;
}

/** API 서버 기준 전체 URL (이미지 등). 상대 경로면 baseURL과 합침. */
export function getApiResourceUrl(pathOrUrl: string): string {
  if (pathOrUrl.startsWith("http")) return pathOrUrl;
  const base = api.defaults.baseURL ?? "";
  if (!base) return pathOrUrl;
  const baseClean = base.replace(/\/$/, "");
  const path = pathOrUrl.startsWith("/") ? pathOrUrl : `/${pathOrUrl}`;
  return `${baseClean}${path}`;
}

/** 학습용 이미지 전체 URL (API base + image_url). */
export function getTrainingImageFullUrl(imageUrl: string): string {
  return getApiResourceUrl(imageUrl);
}

export function isApiError(error: unknown): error is AxiosError<ErrorDetail> {
  return axios.isAxiosError(error) && error.response?.data !== undefined;
}

export function getErrorMessage(error: unknown): string {
  if (axios.isAxiosError(error)) {
    if (error.response?.status === 504)
      return "게이트웨이 타임아웃(504)입니다. 프록시/Ingress의 read_timeout을 늘리거나, 스트리밍이 끊기지 않도록 해당 경로에 버퍼링 끄기를 적용해 주세요.";
    if (error.code === "ECONNABORTED" || (error.message && /timeout/i.test(error.message)))
      return "응답 시간이 초과되었습니다. 잠시 후 다시 시도해 주세요.";
    if (error.code === "ERR_NETWORK" || (error.message && /network error/i.test(error.message)))
      return "연결이 끊겼습니다. (스트리밍 끊김 후 대체 요청이 504이면 CORS가 막혀 보일 수 있습니다. 게이트웨이에서 스트리밍 경로 버퍼링 끄기 및 타임아웃·CORS 설정을 확인하세요.)";
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
