import axios, { AxiosError } from "axios";
import type {
  GenerateResponse,
  ErrorDetail,
  StylesResponse,
  TrainingItem,
  ACBiologicalAnalysis,
  ACReconstructRequest,
  ImageAnalysisResponse,
  ViewpointAnalysisResponse,
  UniversalAnalysisResponse,
} from "../types/api";

/** 프론트 요청 타임아웃: 2분 (이미지 생성 등 긴 API용). 동영상/댄스는 타임아웃 없음(무제한). */
const FRONTEND_TIMEOUT_MS = 2 * 60 * 1000; // 120_000

/**
 * 백엔드 베이스 URL.
 * 비우면 현재 페이지 origin 기준 상대 경로(/api, /health) — Vite 프록시(3000→7000) 또는 백엔드가 정적 파일을 같이 서빙할 때 동작.
 * 원격 API만 쓸 때는 .env에 VITE_API_BASE_URL=https://your-api.example.com 설정.
 */
const DEFAULT_API_BASE = "";
const getBaseURL = (): string => {
  const url = (import.meta as { env?: { VITE_API_BASE_URL?: string } }).env?.VITE_API_BASE_URL ?? "";
  return (url || "").trim() || DEFAULT_API_BASE;
};

const api = axios.create({
  baseURL: getBaseURL() || undefined,
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
  baseURL: api.defaults.baseURL ?? undefined,
  timeout: FRONTEND_TIMEOUT_MS,
});
uploadApi.interceptors.request.use((config) => {
  if (config.data instanceof FormData && config.headers) {
    delete (config.headers as Record<string, unknown>)["Content-Type"];
  }
  return config;
});

const IMAGE_POLL_INTERVAL_MS = 3000;
const IMAGE_POLL_MAX_WAIT_MS = 8 * 60 * 1000; // 포즈 보강·재시도 시 여유

/** Z-Image / SDXL: 비동기 잡 제출 후 폴링 (62초+ 걸릴 때 타임아웃 회피). species: 강아지/고양이에 맞는 프롬프트 적용. */
export async function generateImage(
  file: File,
  style: string,
  customPrompt: string | null,
  strength: number | null,
  seed: number | null,
  sideProfileLock?: boolean,
  options?: {
    species?: "dog" | "cat" | null;
    usePoseLock?: boolean;
    analysis?: UniversalAnalysisResponse | Record<string, unknown>;
    validateAndRetry?: boolean;
  }
): Promise<GenerateResponse> {
  const form = new FormData();
  form.append("style", style);
  form.append("image", file);
  form.append("custom_prompt", (customPrompt ?? "").trim());
  form.append("strength", String(strength ?? 0.5));
  if (seed !== null) form.append("seed", String(seed));
  if (sideProfileLock) form.append("side_profile_lock", "true");
  if (options?.species) form.append("species", options.species);
  if (options?.usePoseLock && options?.analysis != null) {
    form.append("use_pose_lock", "true");
    form.append("analysis", JSON.stringify(options.analysis));
  }
  if (options?.validateAndRetry) form.append("validate_and_retry", "true");

  // 1) 비동기 잡 제출 (즉시 job_id 반환)
  let jobId: string;
  try {
    const { data: jobResp } = await uploadApi.post<{ job_id: string }>("/api/image/generate", form);
    jobId = jobResp.job_id;
    if (!jobId) throw new Error("No job_id from server");
  } catch (err: unknown) {
    // 구백엔드: /api/image/generate 없으면 404 → 기존 동기 /api/generate 로 폴백
    const ax = err as { response?: { status?: number } };
    if (ax?.response?.status === 404) {
      const { data } = await uploadApi.post<{
        original_url?: string;
        generated_url?: string;
        image_url?: string;
        processing_time?: number;
        processing_time_seconds?: number;
        generated_image_base64?: string | null;
      }>("/api/generate", form);
      return {
        original_url: data.original_url ?? "",
        generated_url: data.generated_url ?? data.image_url ?? "",
        processing_time: data.processing_time ?? data.processing_time_seconds ?? 0,
        generated_image_base64: data.generated_image_base64 ?? null,
      };
    }
    throw err;
  }

  // 2) 완료될 때까지 폴링
  const started = Date.now();
  while (Date.now() - started < IMAGE_POLL_MAX_WAIT_MS) {
    const { data: status } = await api.get<{
      status: string;
      original_url?: string | null;
      generated_url?: string | null;
      processing_time?: number | null;
      generated_image_base64?: string | null;
      error?: string | null;
    }>(`/api/image/status/${jobId}`);
    if (status.status === "completed") {
      return {
        original_url: status.original_url ?? "",
        generated_url: status.generated_url ?? "",
        processing_time: status.processing_time ?? 0,
        generated_image_base64: status.generated_image_base64 ?? null,
      };
    }
    if (status.status === "failed") {
      throw new Error(status.error ?? "Image generation failed");
    }
    await new Promise((r) => setTimeout(r, IMAGE_POLL_INTERVAL_MS));
  }
  throw new Error("이미지 생성 시간이 초과되었습니다. 다시 시도해 주세요.");
}

/** API에 없을 때 사용할 스타일 기본 표시 (구버전 백엔드 대응) */
const FALLBACK_STYLES: StylesResponse = {
  sailor_moon: "Sailor Moon (magical girl, sparkle)",
  pixel_art: "Pixel Art (sprite, 16 colors)",
  animal_crossing: "게임 캐릭터 (구조 재디자인)",
  clay_art: "클레이 아트 (손수 제작 점토 조각)",
};

export async function getStyles(): Promise<StylesResponse> {
  const { data } = await api.get<StylesResponse>("/api/styles");
  if (!data || typeof data !== "object") return FALLBACK_STYLES;
  if (Object.keys(data).length === 0) return FALLBACK_STYLES;
  const missing =
    !data.sailor_moon || !data.pixel_art || !data.animal_crossing || !data.clay_art;
  if (missing) return { ...data, ...FALLBACK_STYLES };
  return data;
}

// ---------- LTX-2 Image-to-Video ----------

export type GenerateVideoResponse = {
  video_url: string;
  processing_time: number;
  video_base64?: string | null;
};

export type VideoPresetsResponse = Record<string, string>;

export async function getVideoPresets(): Promise<VideoPresetsResponse> {
  const { data } = await api.get<VideoPresetsResponse>("/api/video/presets");
  return data;
}

/** 백엔드에서 ComfyUI LTX 워크플로 JSON 가져오기. 성공 시 동영상 생성은 백엔드 프록시(/api/video/comfyui/*) 경유. */
export async function getComfyUIWorkflow(): Promise<Record<string, unknown>> {
  const { data } = await api.get<Record<string, unknown>>("/api/video/comfyui/workflow");
  return data;
}

/** 워크플로에 이미지 파일명·프롬프트 주입 (ComfyUI 노드: LoadImage, CLIPTextEncode 등). */
function injectWorkflowInputs(
  workflow: Record<string, { class_type?: string; inputs?: Record<string, unknown> }>,
  imageName: string,
  promptText: string
): Record<string, { class_type?: string; inputs?: Record<string, unknown> }> {
  const out: Record<string, { class_type?: string; inputs?: Record<string, unknown> }> = {};
  for (const [nid, node] of Object.entries(workflow)) {
    if (!node || typeof node !== "object") {
      out[nid] = node;
      continue;
    }
    const ct = ((node.class_type as string) ?? "").toLowerCase();
    const inputs = { ...(node.inputs ?? {}) };
    if (ct.includes("loadimage") || ct.includes("load image")) {
      inputs.image = imageName;
    }
    if (ct.includes("clip") && ct.includes("text")) {
      inputs.text = promptText;
    }
    out[nid] = { ...node, inputs };
  }
  return out;
}

const VIDEO_POLL_INTERVAL_MS = 5000;
const VIDEO_POLL_MAX_WAIT_MS = 30 * 60 * 1000; // 30분
const COMFYUI_POLL_INTERVAL_MS = 2000;
const COMFYUI_POLL_MAX_MS = 30 * 60 * 1000;

/** 대용량 ArrayBuffer → base64 (spread/btoa 한 번에 넣으면 스택 초과 가능) */
function arrayBufferToBase64(buf: ArrayBuffer): string {
  const bytes = new Uint8Array(buf);
  const chunk = 0x8000;
  let binary = "";
  for (let i = 0; i < bytes.length; i += chunk) {
    binary += String.fromCharCode.apply(null, bytes.subarray(i, i + chunk) as unknown as number[]);
  }
  return btoa(binary);
}

/** ComfyUI 동영상 생성 (업로드/prompt/history/view는 백엔드 프록시 경로로 요청). */
async function generateVideoViaComfyUI(
  file: File,
  prompt: string,
  _preset?: string | null,
  _negativePrompt?: string | null,
  workflowRaw?: Record<string, unknown>
): Promise<GenerateVideoResponse> {
  const backendBase = (getBaseURL() ?? "").replace(/\/$/, "");
  const raw = workflowRaw ?? (await getComfyUIWorkflow());
  const promptMap =
    raw?.prompt && typeof raw.prompt === "object" && !Array.isArray(raw.prompt)
      ? (raw.prompt as Record<string, { class_type?: string; inputs?: Record<string, unknown> }>)
      : (raw as Record<string, { class_type?: string; inputs?: Record<string, unknown> }>);
  if (!promptMap || Object.keys(promptMap).length === 0) {
    throw new Error("Invalid ComfyUI workflow: no prompt map");
  }

  const form = new FormData();
  form.append("image", file);
  const uploadRes = await fetch(`${backendBase}/api/video/comfyui/upload/image`, {
    method: "POST",
    body: form,
    credentials: "include",
  });
  if (!uploadRes.ok) {
    const t = await uploadRes.text();
    throw new Error(`ComfyUI upload failed: ${uploadRes.status} ${t.slice(0, 200)}`);
  }
  const uploadJson = (await uploadRes.json()) as { name?: string; subfolder?: string };
  const imageName = uploadJson.name ?? "image.png";

  const injected = injectWorkflowInputs(promptMap, imageName, (prompt ?? "").trim() || "gentle motion");
  const promptId = crypto.randomUUID?.() ?? "xxxxxxxx-xxxx-4xxx-yxxx-xxxxxxxxxxxx".replace(/[xy]/g, () => ((Math.random() * 16) | 0).toString(16));
  const promptRes = await fetch(`${backendBase}/api/video/comfyui/prompt`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ prompt: injected, prompt_id: promptId }),
    credentials: "include",
  });
  if (!promptRes.ok) {
    const t = await promptRes.text();
    throw new Error(`ComfyUI prompt failed: ${promptRes.status} ${t.slice(0, 300)}`);
  }
  const promptJson = (await promptRes.json()) as { error?: string; prompt_id?: string };
  if (promptJson.error) throw new Error(`ComfyUI: ${promptJson.error}`);
  const pid = promptJson.prompt_id ?? promptId;

  const start = Date.now();
  let videoFilename: string | null = null;
  let subfolder = "";
  let typeOut = "output";
  while (Date.now() - start < COMFYUI_POLL_MAX_MS) {
    const hisRes = await fetch(`${backendBase}/api/video/comfyui/history/${pid}`, { credentials: "include" });
    if (!hisRes.ok) {
      await new Promise((r) => setTimeout(r, COMFYUI_POLL_INTERVAL_MS));
      continue;
    }
    const his = (await hisRes.json()) as Record<string, unknown>;
    const node = (typeof his[pid] === "object" && his[pid] !== null ? his[pid] : his) as Record<string, unknown>;
    const outputs = (node?.outputs ?? {}) as Record<string, { videos?: Array<{ filename?: string; subfolder?: string; type?: string }>; gifs?: unknown[]; animations?: unknown[] }>;
    for (const _nid of Object.keys(outputs)) {
      const out = outputs[_nid];
      const videos = out?.videos ?? out?.gifs ?? out?.animations;
      if (Array.isArray(videos) && videos.length > 0) {
        const v = videos[0] as { filename?: string; subfolder?: string; type?: string };
        if (v?.filename) {
          videoFilename = v.filename;
          subfolder = v.subfolder ?? "";
          typeOut = v.type ?? "output";
          break;
        }
      }
    }
    if (videoFilename) break;
    await new Promise((r) => setTimeout(r, COMFYUI_POLL_INTERVAL_MS));
  }
  if (!videoFilename) throw new Error("ComfyUI video generation timed out or no video output");

  const viewParams = new URLSearchParams({ filename: videoFilename, type: typeOut });
  if (subfolder) viewParams.set("subfolder", subfolder);
  const viewRes = await fetch(`${backendBase}/api/video/comfyui/view?${viewParams.toString()}`, { credentials: "include" });
  if (!viewRes.ok) throw new Error(`ComfyUI view failed: ${viewRes.status}`);
  const videoBlob = await viewRes.blob();
  const processingTime = (Date.now() - start) / 1000;
  const buf = await videoBlob.arrayBuffer();
  const b64 = arrayBufferToBase64(buf);
  return {
    video_url: "",
    processing_time: processingTime,
    video_base64: b64,
  };
}

export async function generateVideo(
  file: File,
  prompt: string,
  preset?: string | null,
  negativePrompt?: string | null
): Promise<GenerateVideoResponse> {
  try {
    const w = await getComfyUIWorkflow();
    if (w && typeof w === "object" && Object.keys(w).length > 0) {
      return await generateVideoViaComfyUI(file, prompt, preset, negativePrompt, w);
    }
  } catch {
    // 워크플로 없음 또는 ComfyUI 비활성 → 백엔드 잡 방식 사용
  }

  const form = new FormData();
  form.append("image", file);
  form.append("prompt", (prompt ?? "").trim());
  if (preset) form.append("preset", preset);
  if (negativePrompt) form.append("negative_prompt", negativePrompt);
  const uploadVideoApi = axios.create({
    baseURL: api.defaults.baseURL ?? undefined,
    timeout: 60 * 1000, // 업로드+job_id 수신만 하므로 1분이면 충분
  });
  uploadVideoApi.interceptors.request.use((config) => {
    if (config.data instanceof FormData && config.headers) {
      delete (config.headers as Record<string, unknown>)["Content-Type"];
    }
    return config;
  });
  const { data: jobResp } = await uploadVideoApi.post<{ job_id: string }>("/api/video/generate", form);
  const jobId = jobResp.job_id;
  if (!jobId) throw new Error("No job_id from server");

  const started = Date.now();
  while (Date.now() - started < VIDEO_POLL_MAX_WAIT_MS) {
    const { data: status } = await api.get<{
      status: string;
      video_url?: string | null;
      processing_time?: number | null;
      video_base64?: string | null;
      error?: string | null;
    }>(`/api/video/status/${jobId}`);
    if (status.status === "completed") {
      return {
        video_url: status.video_url ?? "",
        processing_time: status.processing_time ?? 0,
        video_base64: status.video_base64 ?? null,
      };
    }
    if (status.status === "failed") {
      throw new Error(status.error ?? "Video generation failed");
    }
    await new Promise((r) => setTimeout(r, VIDEO_POLL_INTERVAL_MS));
  }
  throw new Error("동영상 생성 시간이 초과되었습니다. 다시 시도해 주세요.");
}

/** Z-Image( main ) 백엔드는 /health, SDXL은 /api/health. 이 앱은 Z-Image 기준이므로 /health 사용 */
export async function getHealth(): Promise<{ status: string; gpu_available: boolean }> {
  const { data } = await api.get<{ status: string; gpu_available: boolean }>("/health");
  return data;
}

// ---------- Dance / Motion Transfer ----------

export type DanceMotionItem = {
  id: string;
  label: string;
  videoReference: string;
};

export type DanceGenerateResponse = {
  video_url: string;
  processing_time: number;
  motion_id: string;
  character: string;
};

const DANCE_POLL_INTERVAL_MS = 2000;
const DANCE_POLL_MAX_WAIT_MS = 30 * 60 * 1000; // 30분

// 댄스 job 폴링 공통 로직
async function pollDanceJob(jobId: string): Promise<DanceGenerateResponse> {
  const started = Date.now();
  while (Date.now() - started < DANCE_POLL_MAX_WAIT_MS) {
    const { data } = await api.get<{
      status: string;
      video_url?: string | null;
      processing_time?: number | null;
      motion_id?: string | null;
      character?: string | null;
      error?: string | null;
    }>(`/api/dance/status/${jobId}`);

    if (data.status === "completed" && data.video_url) {
      return {
        video_url: data.video_url,
        processing_time: data.processing_time ?? 0,
        motion_id: data.motion_id ?? "",
        character: data.character ?? "",
      };
    }
    if (data.status === "failed") {
      throw new Error(data.error ?? "Dance generation failed");
    }
    await new Promise((r) => setTimeout(r, DANCE_POLL_INTERVAL_MS));
  }
  throw new Error("Dance generation timed out");
}

export async function getDanceMotions(): Promise<DanceMotionItem[]> {
  const { data } = await api.get<DanceMotionItem[]>("/api/dance/motions");
  return Array.isArray(data) ? data : [];
}

/** 사전 등록 댄스 영상 한 건 (GET /api/dance/list) */
export type DanceVideoInfo = {
  id: string;
  display_name: string;
  filename: string;
  duration_seconds: number;
  fps: number;
  width: number;
  height: number;
  frame_count: number;
  file_size_mb: number;
};

export type DanceListResponse = {
  total: number;
  dances: DanceVideoInfo[];
};

/** 서버 폴더(motions/ 등)에 등록된 댄스 영상 목록 */
export async function getDanceList(): Promise<DanceListResponse> {
  const { data } = await api.get<DanceListResponse>("/api/dance/list");
  return { total: data?.total ?? 0, dances: Array.isArray(data?.dances) ? data.dances : [] };
}

/** 댄스 폴더 재스캔 후 목록 갱신 */
export async function refreshDanceList(): Promise<{ message: string; added: string[]; total: number }> {
  const { data } = await api.post<{ message: string; added: string[]; total: number }>("/api/dance/refresh");
  return data;
}

/** 댄스 생성 파이프라인: ltx(기본)=LTX+레퍼런스 영상, pose_sdxl=포즈→ComfyUI 프레임→ffmpeg */
export type DancePipelineMode = "ltx" | "pose_sdxl";

export async function generateDance(
  file: File,
  motionId: string,
  character: "dog" | "cat",
  pipeline: DancePipelineMode = "ltx"
): Promise<DanceGenerateResponse> {
  const form = new FormData();
  form.append("image", file);
  form.append("motion_id", motionId);
  form.append("character", character);
  form.append("pipeline", pipeline);
  const uploadDanceApi = axios.create({
    baseURL: api.defaults.baseURL ?? undefined,
    timeout: 60 * 1000, // 파일 업로드 + job_id 수신만 — 1분이면 충분
  });
  uploadDanceApi.interceptors.request.use((config) => {
    if (config.data instanceof FormData && config.headers) {
      delete (config.headers as Record<string, unknown>)["Content-Type"];
    }
    return config;
  });
  const { data: jobResp } = await uploadDanceApi.post<{ job_id: string }>("/api/dance/generate", form);
  if (!jobResp.job_id) throw new Error("No job_id from server");
  return pollDanceJob(jobResp.job_id);
}

export async function generateDanceCustom(
  characterImage: File,
  referenceVideo: File,
  character: "dog" | "cat",
  pipeline: DancePipelineMode = "ltx"
): Promise<DanceGenerateResponse> {
  const form = new FormData();
  form.append("image", characterImage);
  form.append("reference_video", referenceVideo);
  form.append("character", character);
  form.append("pipeline", pipeline);
  const uploadDanceApi = axios.create({
    baseURL: api.defaults.baseURL ?? undefined,
    timeout: 60 * 1000, // 파일 업로드 + job_id 수신만
  });
  uploadDanceApi.interceptors.request.use((config) => {
    if (config.data instanceof FormData && config.headers) {
      delete (config.headers as Record<string, unknown>)["Content-Type"];
    }
    return config;
  });
  const { data: jobResp } = await uploadDanceApi.post<{ job_id: string }>("/api/dance/generate-custom", form);
  if (!jobResp.job_id) throw new Error("No job_id from server");
  return pollDanceJob(jobResp.job_id);
}

// ---------- AC Villager Reconstruction (Stage 1 + Stage 2) ----------

/** Stage 1: biological analysis. Optional image + form overrides. Returns structured data only. */
export async function acAnalyze(params: {
  file?: File | null;
  species?: string | null;
  main_fur_color?: string | null;
  secondary_fur_color?: string | null;
  eye_color?: string | null;
  markings?: string | null;
  ear_type?: string | null;
  tail_type?: string | null;
}): Promise<ACBiologicalAnalysis> {
  const form = new FormData();
  if (params.file) form.append("image", params.file);
  if (params.species) form.append("species", params.species);
  if (params.main_fur_color != null) form.append("main_fur_color", params.main_fur_color);
  if (params.secondary_fur_color != null) form.append("secondary_fur_color", params.secondary_fur_color);
  if (params.eye_color != null) form.append("eye_color", params.eye_color);
  if (params.markings != null) form.append("markings", params.markings);
  if (params.ear_type != null) form.append("ear_type", params.ear_type);
  if (params.tail_type != null) form.append("tail_type", params.tail_type);
  const { data } = await uploadApi.post<ACBiologicalAnalysis>("/api/ac/analyze", form);
  return data;
}

/** Universal analysis: pose, camera, gravity, clothing, structure. JSON only. */
export async function analyzeUniversal(params?: {
  file?: File | null;
  species?: string | null;
  view_angle?: string | null;
  body_pose?: string | null;
  gravity_axis?: string | null;
  head_direction_degrees?: number | null;
  spine_alignment?: string | null;
  visible_eyes?: number | null;
  leg_visibility_count?: number | null;
  is_full_body_visible?: boolean | null;
  is_wearing_clothes?: boolean | null;
  clothing_type?: string | null;
  clothing_color?: string | null;
  clothing_pattern?: string | null;
  clothing_confidence?: number | null;
}): Promise<UniversalAnalysisResponse> {
  const form = new FormData();
  if (params?.file) form.append("image", params.file);
  if (params?.species != null) form.append("species", params.species);
  if (params?.view_angle != null) form.append("view_angle", params.view_angle);
  if (params?.body_pose != null) form.append("body_pose", params.body_pose);
  if (params?.gravity_axis != null) form.append("gravity_axis", params.gravity_axis);
  if (params?.head_direction_degrees != null) form.append("head_direction_degrees", String(params.head_direction_degrees));
  if (params?.spine_alignment != null) form.append("spine_alignment", params.spine_alignment);
  if (params?.visible_eyes != null) form.append("visible_eyes", String(params.visible_eyes));
  if (params?.leg_visibility_count != null) form.append("leg_visibility_count", String(params.leg_visibility_count));
  if (params?.is_full_body_visible != null) form.append("is_full_body_visible", String(params.is_full_body_visible));
  if (params?.is_wearing_clothes != null) form.append("is_wearing_clothes", String(params.is_wearing_clothes));
  if (params?.clothing_type != null) form.append("clothing_type", params.clothing_type);
  if (params?.clothing_color != null) form.append("clothing_color", params.clothing_color);
  if (params?.clothing_pattern != null) form.append("clothing_pattern", params.clothing_pattern);
  if (params?.clothing_confidence != null) form.append("clothing_confidence", String(params.clothing_confidence));
  const { data } = await uploadApi.post<UniversalAnalysisResponse>("/api/image/analyze-universal", form);
  return data;
}

/** Viewpoint analysis: camera angle and subject orientation. JSON only. */
export async function analyzeViewpoint(params?: {
  file?: File | null;
  view_angle?: string | null;
  head_visible_eyes?: number | null;
  body_orientation_degrees?: number | null;
  tail_visible?: boolean | null;
}): Promise<ViewpointAnalysisResponse> {
  const form = new FormData();
  if (params?.file) form.append("image", params.file);
  if (params?.view_angle != null) form.append("view_angle", params.view_angle);
  if (params?.head_visible_eyes != null) form.append("head_visible_eyes", String(params.head_visible_eyes));
  if (params?.body_orientation_degrees != null) form.append("body_orientation_degrees", String(params.body_orientation_degrees));
  if (params?.tail_visible != null) form.append("tail_visible", String(params.tail_visible));
  const { data } = await uploadApi.post<ViewpointAnalysisResponse>("/api/image/viewpoint", form);
  return data;
}

/** Image analysis: structured visual attributes (animal, clothing, accessories, pose, environment). JSON only. */
export async function analyzeImage(params: {
  file?: File | null;
  species?: string | null;
  fur_main_color?: string | null;
  fur_secondary_color?: string | null;
  major_markings?: string | null;
  is_wearing_clothes?: boolean | null;
  clothing_type?: string | null;
  clothing_color?: string | null;
  posture?: string | null;
}): Promise<ImageAnalysisResponse> {
  const form = new FormData();
  if (params.file) form.append("image", params.file);
  if (params.species != null) form.append("species", params.species);
  if (params.fur_main_color != null) form.append("fur_main_color", params.fur_main_color);
  if (params.fur_secondary_color != null) form.append("fur_secondary_color", params.fur_secondary_color);
  if (params.major_markings != null) form.append("major_markings", params.major_markings);
  if (params.is_wearing_clothes != null) form.append("is_wearing_clothes", String(params.is_wearing_clothes));
  if (params.clothing_type != null) form.append("clothing_type", params.clothing_type);
  if (params.clothing_color != null) form.append("clothing_color", params.clothing_color);
  if (params.posture != null) form.append("posture", params.posture);
  const { data } = await uploadApi.post<ImageAnalysisResponse>("/api/image/analyze", form);
  return data;
}

/** Stage 2: villager reconstruction (T2I only). No image; uses biological data. */
export async function acReconstruct(body: ACReconstructRequest): Promise<GenerateResponse> {
  const payload = {
    species: body.species,
    main_fur_color: body.main_fur_color ?? "cream",
    secondary_fur_color: body.secondary_fur_color ?? "none",
    eye_color: body.eye_color ?? "amber",
    markings: body.markings ?? "none",
    ear_type: body.ear_type ?? null,
    tail_type: body.tail_type ?? null,
    seed: body.seed ?? null,
  };
  const { data } = await api.post<GenerateResponse>("/api/generate/ac-villager-reconstruct", payload);
  return {
    original_url: data.original_url ?? "",
    generated_url: data.generated_url ?? "",
    processing_time: data.processing_time ?? 0,
    generated_image_base64: data.generated_image_base64 ?? null,
  };
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

/** 불완전한 JSON(시작만 있고 닫는 ``` 없음)이면 true. 저장 시 대체 문구 쓰기 위함 */
export function isTruncatedStructuredContent(content: string): boolean {
  if (!content?.trim()) return false;
  const hasOpen = /```\s*json\s*/i.test(content);
  const hasClose = /```\s*$/.test(content.trim()) || content.trim().endsWith("```");
  return hasOpen && !hasClose && content.length > 200;
}

/** 스트리밍 응답 본문에서 ```json ... ``` 블록을 파싱해 구조화 데이터 반환. 실패 시 null. 잘린 응답·블록 없음도 보완 후 파싱 시도. */
export function parseStructuredFromContent(content: string): HealthChatStructured | null {
  const raw = content?.trim();
  if (!raw) return null;

  const tryParse = (str: string): HealthChatStructured | null => {
    try {
      const data = JSON.parse(str) as Record<string, unknown>;
      if (!Array.isArray(data.differential) || data.differential.length === 0) return null;
      return data as unknown as HealthChatStructured;
    } catch {
      return null;
    }
  };

  const tryRepair = (jsonStr: string): HealthChatStructured | null => {
    let out = tryParse(jsonStr);
    if (out) return out;
    let trimmed = jsonStr.trim().replace(/,?\s*"[^"]*$/, "").replace(/,?\s*$/, "");
    const suffixes = ["]", "}", "]}", "}]", "}]}", "]}]", "}]]}"];
    for (const suf of suffixes) {
      out = tryParse(trimmed + suf);
      if (out) return out;
    }
    return null;
  };

  // 1) 정상 블록: ```json ... ```
  const match = raw.match(/```\s*json\s*([\s\S]*?)```/i);
  if (match?.[1]) {
    const out = tryRepair(match[1].trim());
    if (out) return out;
  }

  // 2) 잘림: ```json 은 있는데 닫는 ``` 없음
  const openMatch = raw.match(/```\s*json\s*([\s\S]*)$/i);
  if (openMatch?.[1]) {
    const out = tryRepair(openMatch[1].trim());
    if (out) return out;
  }

  // 3) 블록 없이 본문에 "differential" 포함된 JSON만 있는 경우 (웹 응답 형식 차이 대응)
  const diffIdx = raw.indexOf('"differential"');
  if (diffIdx !== -1) {
    let start = raw.lastIndexOf("{", diffIdx);
    if (start !== -1) {
      const candidate = raw.slice(start);
      const out = tryRepair(candidate);
      if (out) return out;
    }
  }

  return null;
}

const HEALTH_CHAT_INTRO =
  "아래 감별 가능성과 집에서 확인할 점을 참고하세요. 정확한 판단은 의료·수의 전문가에게 확인하세요.";

/** 구조화 응답일 때 사용할 짧은 안내 문구 (사고 과정 노출 방지) */
export const HEALTH_CHAT_SHORT_INTRO = HEALTH_CHAT_INTRO;

const getBaseUrl = () => {
  const u = (import.meta as { env?: { VITE_API_BASE_URL?: string } }).env?.VITE_API_BASE_URL?.trim();
  return u || DEFAULT_API_BASE;
};

/** 스트리밍 채팅: 청크마다 onChunk 호출, 완료 시 전체 content 반환. 표시가 빨리 되고 체감 속도 개선. */
export async function llmChatStream(
  messages: Array<{ role: string; content: string }>,
  onChunk: (chunk: string) => void,
  maxTokens = 4096,
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
  maxTokens = 4096,
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

/** 재생용: 백엔드는 순수 base64; 예전 클라이언트가 data: 를 중복 붙인 경우에도 재생 가능 */
export function videoSrcFromApiField(videoBase64: string | null | undefined, videoUrl: string): string {
  if (!videoBase64?.trim()) return getApiResourceUrl(videoUrl);
  const t = videoBase64.trim();
  if (t.startsWith("data:")) return t;
  return `data:video/mp4;base64,${t}`;
}

/** 다운로드용: data URL에서 raw base64만 추출 */
export function rawBase64FromVideoField(videoBase64: string | null | undefined): string | null {
  if (!videoBase64?.trim()) return null;
  let t = videoBase64.trim();
  const idx = t.indexOf("base64,");
  if (t.startsWith("data:") && idx !== -1) t = t.slice(idx + 7);
  return t;
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
  if (error instanceof Error) {
    const msg = error.message;
    if (/Dance generation timed out|timed out/i.test(msg))
      return "영상 생성 대기 시간이 초과되었습니다. 서버가 바쁠 수 있으니 잠시 후 다시 시도해 주세요.";
    return msg;
  }
  return "오류가 발생했습니다.";
}
