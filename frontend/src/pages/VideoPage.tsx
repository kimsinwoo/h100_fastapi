import { useCallback, useEffect, useState } from "react";
import { Link } from "react-router-dom";
import { ImageUploader } from "../components/ImageUploader";
import { LoadingOverlay } from "../components/LoadingSpinner";
import {
  generateVideo,
  getVideoPresets,
  getHealth,
  getErrorMessage,
  getApiResourceUrl,
  getDanceMotions,
  generateDance,
} from "../services/api";
import type {
  GenerateVideoResponse,
  VideoPresetsResponse,
  DanceMotionItem,
} from "../services/api";

type VideoState =
  | { phase: "idle" }
  | { phase: "loading" }
  | { phase: "success"; data: GenerateVideoResponse }
  | { phase: "error"; message: string };

const DEFAULT_PROMPT = "The character smiles and slowly turns their head toward the camera.";

export default function VideoPage() {
  const [file, setFile] = useState<File | null>(null);
  const [prompt, setPrompt] = useState<string>(DEFAULT_PROMPT);
  const [presets, setPresets] = useState<VideoPresetsResponse | null>(null);
  const [state, setState] = useState<VideoState>({ phase: "idle" });
  const [gpuAvailable, setGpuAvailable] = useState<boolean | null>(null);
  const [danceMotions, setDanceMotions] = useState<DanceMotionItem[]>([]);
  const [selectedMotionId, setSelectedMotionId] = useState<string>("rat_dance");
  const [danceCharacter, setDanceCharacter] = useState<"dog" | "cat">("dog");

  useEffect(() => {
    getHealth()
      .then((r) => setGpuAvailable(r.gpu_available))
      .catch(() => setGpuAvailable(null));
    getVideoPresets()
      .then(setPresets)
      .catch(() => setPresets({ smile_turn: "", wind_leaves: "", dancing_pet: "", golden_hour: "", cozy_moment: "", neon_night: "", dreamy_bokeh: "" }));
    getDanceMotions()
      .then(setDanceMotions)
      .catch(() => setDanceMotions([]));
  }, []);

  const handlePreset = useCallback(
    (key: string) => {
      const text = presets?.[key];
      if (text) setPrompt(text);
    },
    [presets]
  );

  const handleGenerate = useCallback(async () => {
    if (!file) return;
    if (!prompt.trim()) {
      setState({ phase: "error", message: "프롬프트를 입력해주세요." });
      return;
    }
    setState({ phase: "loading" });
    try {
      const data = await generateVideo(file, prompt.trim(), null, null);
      setState({ phase: "success", data });
    } catch (err) {
      setState({ phase: "error", message: getErrorMessage(err) });
    }
  }, [file, prompt]);

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
        <LoadingOverlay message="동영상 생성 중... (LTX-2, 1~3분 소요될 수 있습니다)" />
      )}

      <div className="mx-auto max-w-4xl px-4">
        <header className="mb-8 flex items-center justify-between">
          <div className="text-center">
            <h1 className="text-3xl font-bold text-gray-900">LTX-2 이미지 → 동영상</h1>
            <p className="mt-2 text-gray-600">사진과 프롬프트로 동영상 생성 (Hugging Face LTX-2)</p>
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
            <h2 className="mb-2 text-sm font-semibold text-gray-700">2. 동영상 프롬프트</h2>
            <p className="mb-2 text-xs text-gray-500">
              원하는 동작·장면을 영어로 설명하세요. 아래 테스트 스타일을 선택해도 됩니다.
            </p>
            {presets && Object.keys(presets).length > 0 && (
              <div className="mb-3 flex flex-wrap gap-2">
                {Object.entries(presets).map(([key]) => (
                  <button
                    key={key}
                    type="button"
                    onClick={() => handlePreset(key)}
                    disabled={isProcessing}
                    className="rounded-lg border-2 border-gray-200 bg-white px-3 py-1.5 text-sm font-medium text-gray-700 hover:border-indigo-400 disabled:opacity-60"
                  >
                    {key === "smile_turn"
                      ? "고개 돌리기"
                      : key === "wind_leaves"
                        ? "바람/나뭇잎"
                        : key === "dancing_pet"
                          ? "춤추는 강아지/고양이"
                          : key === "golden_hour"
                            ? "황금시간대"
                            : key === "cozy_moment"
                              ? "코지 모먼트"
                              : key === "neon_night"
                                ? "네온 야경"
                                : key === "dreamy_bokeh"
                                  ? "드리밍 보케"
                                  : key}
                  </button>
                ))}
              </div>
            )}
            <textarea
              value={prompt}
              onChange={(e) => setPrompt(e.target.value)}
              disabled={isProcessing}
              placeholder="e.g. The character smiles and slowly turns their head."
              rows={3}
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

          <section className="border-t border-gray-200 pt-6">
            <h2 className="mb-2 text-sm font-semibold text-gray-700">Dance Style (Pose 기반 댄스)</h2>
            <p className="mb-3 text-xs text-gray-500">
              참조 영상의 안무를 추출해 캐릭터가 따라하는 영상을 생성합니다. 위에서 업로드한 이미지를 캐릭터로 사용합니다.
            </p>
            {danceMotions.length > 0 && (
              <div className="mb-3 flex flex-wrap gap-2">
                {danceMotions.map((m) => (
                  <button
                    key={m.id}
                    type="button"
                    onClick={() => setSelectedMotionId(m.id)}
                    disabled={isProcessing}
                    className={`rounded-lg border-2 px-3 py-1.5 text-sm font-medium disabled:opacity-60 ${
                      selectedMotionId === m.id
                        ? "border-indigo-500 bg-indigo-50 text-indigo-700"
                        : "border-gray-200 bg-white text-gray-700 hover:border-indigo-400"
                    }`}
                  >
                    {m.label}
                  </button>
                ))}
              </div>
            )}
            <div className="mb-3 flex items-center gap-4">
              <label className="flex items-center gap-2 text-sm text-gray-700">
                <span>캐릭터</span>
                <select
                  value={danceCharacter}
                  onChange={(e) => setDanceCharacter(e.target.value as "dog" | "cat")}
                  disabled={isProcessing}
                  className="rounded border border-gray-300 px-2 py-1 text-sm"
                >
                  <option value="dog">강아지</option>
                  <option value="cat">고양이</option>
                </select>
              </label>
            </div>
            <button
              type="button"
              onClick={handleDanceGenerate}
              disabled={!canDanceGenerate}
              className="w-full rounded-lg bg-emerald-600 px-4 py-3 font-medium text-white hover:bg-emerald-700 disabled:cursor-not-allowed disabled:opacity-50"
            >
              Dance Style 생성 (RAT 등)
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
