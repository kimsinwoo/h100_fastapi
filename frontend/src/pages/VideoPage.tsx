import { useCallback, useEffect, useState } from "react";
import { Link } from "react-router-dom";
import { ImageUploader } from "../components/ImageUploader";
import { LoadingOverlay } from "../components/LoadingSpinner";
import {
  generateVideo,
  getHealth,
  getErrorMessage,
  getApiResourceUrl,
  getDanceMotions,
  generateDance,
} from "../services/api";
import type { GenerateVideoResponse, DanceMotionItem } from "../services/api";
import {
  VIDEO_PRESETS,
  VIDEO_PRESETS_TOP_20,
  VIDEO_PRESETS_VIRAL,
  NATURAL_PROMPT_TIPS,
  type VideoPresetItem,
} from "../constants/videoPresets";

type VideoState =
  | { phase: "idle" }
  | { phase: "loading" }
  | { phase: "success"; data: GenerateVideoResponse }
  | { phase: "error"; message: string };

type PresetGroup = "basic" | "top20" | "viral";

const PRESET_GROUPS: { key: PresetGroup; label: string; list: VideoPresetItem[] }[] = [
  { key: "basic", label: "기본 8개 (안정적)", list: VIDEO_PRESETS },
  { key: "top20", label: "TOP 20", list: VIDEO_PRESETS_TOP_20 },
  { key: "viral", label: "바이럴 쇼츠", list: VIDEO_PRESETS_VIRAL },
];

const defaultPreset = VIDEO_PRESETS[0];

export default function VideoPage() {
  const [file, setFile] = useState<File | null>(null);
  const [prompt, setPrompt] = useState<string>(defaultPreset.prompt);
  const [negativePrompt, setNegativePrompt] = useState<string>(defaultPreset.negative);
  const [presetGroup, setPresetGroup] = useState<PresetGroup>("basic");
  const [state, setState] = useState<VideoState>({ phase: "idle" });
  const [gpuAvailable, setGpuAvailable] = useState<boolean | null>(null);
  const [danceMotions, setDanceMotions] = useState<DanceMotionItem[]>([]);
  const [selectedMotionId, setSelectedMotionId] = useState<string>("rat_dance");
  const [danceCharacter, setDanceCharacter] = useState<"dog" | "cat">("dog");
  const [showTips, setShowTips] = useState(false);

  const currentPresets = PRESET_GROUPS.find((g) => g.key === presetGroup)?.list ?? VIDEO_PRESETS;

  useEffect(() => {
    getHealth()
      .then((r) => setGpuAvailable(r.gpu_available))
      .catch(() => setGpuAvailable(null));
    getDanceMotions()
      .then(setDanceMotions)
      .catch(() => setDanceMotions([]));
  }, []);

  const handlePreset = useCallback((preset: VideoPresetItem) => {
    setPrompt(preset.prompt);
    setNegativePrompt(preset.negative);
  }, []);

  const handlePresetGroupChange = useCallback((key: PresetGroup) => {
    setPresetGroup(key);
    const group = PRESET_GROUPS.find((g) => g.key === key);
    if (group?.list[0]) {
      setPrompt(group.list[0].prompt);
      setNegativePrompt(group.list[0].negative);
    }
  }, []);

  const handleGenerate = useCallback(async () => {
    if (!file) return;
    if (!prompt.trim()) {
      setState({ phase: "error", message: "프롬프트를 입력해주세요." });
      return;
    }
    setState({ phase: "loading" });
    try {
      const data = await generateVideo(file, prompt.trim(), null, negativePrompt.trim() || null);
      setState({ phase: "success", data });
    } catch (err) {
      setState({ phase: "error", message: getErrorMessage(err) });
    }
  }, [file, prompt, negativePrompt]);

  const handleDanceGenerate = useCallback(async () => {
    if (!file) {
      setState({ phase: "error", message: "캐릭터 이미지를 업로드해주세요." });
      return;
    }
    setState({ phase: "loading" });
    try {
      const data = await generateDance(file, selectedMotionId, danceCharacter);
      setState({
        phase: "success",
        data: {
          video_url: data.video_url,
          processing_time: data.processing_time,
          video_base64: null,
        },
      });
    } catch (err) {
      setState({ phase: "error", message: getErrorMessage(err) });
    }
  }, [file, selectedMotionId, danceCharacter]);

  const videoSrc =
    state.phase === "success"
      ? state.data.video_base64
        ? `data:video/mp4;base64,${state.data.video_base64}`
        : getApiResourceUrl(state.data.video_url)
      : "";

  const handleDownload = useCallback(() => {
    if (state.phase !== "success") return;
    const b64 = state.data.video_base64;
    if (b64) {
      try {
        const bin = atob(b64);
        const bytes = new Uint8Array(bin.length);
        for (let i = 0; i < bin.length; i++) bytes[i] = bin.charCodeAt(i);
        const blob = new Blob([bytes], { type: "video/mp4" });
        const a = document.createElement("a");
        a.href = URL.createObjectURL(blob);
        a.download = `generated-video-${Date.now()}.mp4`;
        a.click();
        URL.revokeObjectURL(a.href);
      } catch {
        setState({ phase: "error", message: "다운로드 실패" });
      }
      return;
    }
    fetch(getApiResourceUrl(state.data.video_url))
      .then((r) => r.blob())
      .then((blob) => {
        const a = document.createElement("a");
        a.href = URL.createObjectURL(blob);
        a.download = `generated-video-${Date.now()}.mp4`;
        a.click();
        URL.revokeObjectURL(a.href);
      })
      .catch(() => setState({ phase: "error", message: "다운로드 실패" }));
  }, [state]);

  const isProcessing = state.phase === "loading";
  const canGenerate = file !== null && prompt.trim().length > 0 && !isProcessing;
  const canDanceGenerate = file !== null && selectedMotionId.length > 0 && !isProcessing;

  return (
    <div className="min-h-screen bg-gray-100 py-8">
      {isProcessing && (
        <LoadingOverlay message="동영상 생성 중... (LTX-2, 5~10분 소요될 수 있습니다. 최대 10분까지 기다려 주세요)" />
      )}

      <div className="mx-auto max-w-4xl px-4">
        <header className="mb-8 flex items-center justify-between">
          <div className="text-center">
            <h1 className="text-3xl font-bold text-gray-900">이미지 → 동영상 (강아지)</h1>
            <p className="mt-2 text-gray-600">주인 시점(POV) · 원본 강아지·배경 유지 (ComfyUI / LTX / Runway / Pika 등)</p>
            {gpuAvailable !== null && (
              <p className="mt-1 text-sm text-gray-500">Backend: {gpuAvailable ? "GPU" : "CPU"}</p>
            )}
          </div>
          <Link
            to="/"
            className="rounded-lg border border-gray-300 bg-white px-4 py-2 text-sm font-medium text-gray-700 hover:bg-gray-50"
          >
            이미지 생성
          </Link>
        </header>

        <div className="space-y-6 rounded-xl bg-white p-6 shadow">
          <section>
            <h2 className="mb-2 text-sm font-semibold text-gray-700">1. 이미지 업로드</h2>
            <ImageUploader onFileSelect={setFile} selectedFile={file} disabled={isProcessing} />
          </section>

          <section>
            <h2 className="mb-2 text-sm font-semibold text-gray-700">2. 동영상 프롬프트 (주인 시점 POV)</h2>
            <p className="mb-2 text-xs text-gray-500">
              카메라=주인 · 강아지만 등장(손만 잠깐 허용) · 원본 강아지·배경 유지. 아래 카테고리에서 프리셋을 선택하거나 직접 수정하세요.
            </p>
            <div className="mb-2 flex flex-wrap gap-2">
              {PRESET_GROUPS.map((g) => (
                <button
                  key={g.key}
                  type="button"
                  onClick={() => handlePresetGroupChange(g.key)}
                  disabled={isProcessing}
                  className={`rounded-lg border-2 px-3 py-1.5 text-sm font-medium disabled:opacity-60 ${
                    presetGroup === g.key
                      ? "border-indigo-500 bg-indigo-50 text-indigo-700"
                      : "border-gray-200 bg-white text-gray-700 hover:border-indigo-400"
                  }`}
                >
                  {g.label}
                </button>
              ))}
            </div>
            <div className="mb-3 flex max-h-32 flex-wrap gap-2 overflow-y-auto rounded-lg border border-gray-200 bg-gray-50/50 p-2">
              {currentPresets.map((p) => (
                <button
                  key={p.id}
                  type="button"
                  onClick={() => handlePreset(p)}
                  disabled={isProcessing}
                  className="rounded-lg border border-gray-200 bg-white px-3 py-1.5 text-xs font-medium text-gray-700 hover:border-indigo-400 hover:bg-indigo-50/50 disabled:opacity-60"
                >
                  {p.label}
                </button>
              ))}
            </div>
            <div className="mb-3">
              <button
                type="button"
                onClick={() => setShowTips((v) => !v)}
                className="text-xs text-indigo-600 hover:underline"
              >
                {showTips ? "자연스러운 프롬프트 팁 접기" : "자연스러운 프롬프트 구조 보기"}
              </button>
              {showTips && (
                <ul className="mt-1 list-inside list-disc space-y-0.5 rounded bg-amber-50/80 p-2 text-xs text-gray-700">
                  {NATURAL_PROMPT_TIPS.map((tip, i) => (
                    <li key={i}>{tip}</li>
                  ))}
                </ul>
              )}
            </div>
            <label className="mb-1 block text-xs font-medium text-gray-600">Positive Prompt</label>
            <textarea
              value={prompt}
              onChange={(e) => setPrompt(e.target.value)}
              disabled={isProcessing}
              placeholder="first-person perspective from the dog's owner..."
              rows={4}
              className="mb-3 w-full rounded-lg border border-gray-300 px-3 py-2 text-sm focus:border-indigo-500 focus:outline-none focus:ring-1 focus:ring-indigo-500 disabled:opacity-60"
            />
            <label className="mb-1 block text-xs font-medium text-gray-600">Negative Prompt (선택)</label>
            <textarea
              value={negativePrompt}
              onChange={(e) => setNegativePrompt(e.target.value)}
              disabled={isProcessing}
              placeholder="person visible, human body, different dog, extra dogs..."
              rows={2}
              className="w-full rounded-lg border border-gray-300 px-3 py-2 text-sm focus:border-indigo-500 focus:outline-none focus:ring-1 focus:ring-indigo-500 disabled:opacity-60"
            />
          </section>

          <section>
            <button
              type="button"
              onClick={handleGenerate}
              disabled={!canGenerate}
              className="w-full rounded-lg bg-indigo-600 px-4 py-3 font-medium text-white hover:bg-indigo-700 disabled:cursor-not-allowed disabled:opacity-50"
            >
              동영상 생성
            </button>
          </section>

          {state.phase === "error" && (
            <div
              role="alert"
              className="rounded-lg border border-red-200 bg-red-50 p-4 text-sm text-red-800"
            >
              {state.message}
            </div>
          )}

          {state.phase === "success" && (
            <section className="border-t border-gray-200 pt-6">
              <h2 className="mb-2 text-sm font-semibold text-gray-700">생성된 동영상</h2>
              <p className="mb-3 text-xs text-gray-500">
                소요 시간: {state.data.processing_time}초
              </p>
              <div className="overflow-hidden rounded-lg border border-gray-200 bg-black">
                <video
                  src={videoSrc}
                  controls
                  className="w-full"
                  playsInline
                />
              </div>
              <button
                type="button"
                onClick={handleDownload}
                className="mt-3 rounded-lg border border-gray-300 bg-white px-4 py-2 text-sm font-medium text-gray-700 hover:bg-gray-50"
              >
                동영상 다운로드
              </button>
            </section>
          )}
        </div>
      </div>
    </div>
  );
}
