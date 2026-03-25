import { useCallback, useEffect, useState } from "react";
import { Link } from "react-router-dom";
import { ImageUploader } from "../components/ImageUploader";
import { LoadingOverlay } from "../components/LoadingSpinner";
import {
  generateVideo,
  getHealth,
  getErrorMessage,
  getApiResourceUrl,
  videoSrcFromApiField,
  rawBase64FromVideoField,
  getDanceMotions,
  getDanceList,
  refreshDanceList,
  generateDance,
  generateDanceCustom,
} from "../services/api";
import type {
  GenerateVideoResponse,
  DanceMotionItem,
  DanceVideoInfo,
  DancePipelineMode,
} from "../services/api";
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

export default function VideoPage() {
  const [file, setFile] = useState<File | null>(null);
  const [prompt, setPrompt] = useState<string>(() => VIDEO_PRESETS[0]?.prompt ?? "");
  const [negativePrompt, setNegativePrompt] = useState<string>(() => VIDEO_PRESETS[0]?.negative ?? "");
  const [presetGroup, setPresetGroup] = useState<PresetGroup>("basic");
  const [state, setState] = useState<VideoState>({ phase: "idle" });
  const [gpuAvailable, setGpuAvailable] = useState<boolean | null>(null);
  const [, setDanceMotions] = useState<DanceMotionItem[]>([]);
  const [danceList, setDanceList] = useState<DanceVideoInfo[]>([]);
  const [selectedMotionId, setSelectedMotionId] = useState<string>("");
  const [danceCharacter, setDanceCharacter] = useState<"dog" | "cat">("dog");
  const [danceListLoading, setDanceListLoading] = useState(false);
  const [danceListRefreshing, setDanceListRefreshing] = useState(false);
  const [showTips, setShowTips] = useState(false);
  const [customRefVideo, setCustomRefVideo] = useState<File | null>(null);
  const [customCharacter, setCustomCharacter] = useState<"dog" | "cat">("dog");
  /** API pipeline: ltx = LTX+레퍼런스, pose_sdxl = 포즈→ComfyUI 프레임 */
  const [dancePipeline, setDancePipeline] = useState<DancePipelineMode>("ltx");
  const [customDancePipeline, setCustomDancePipeline] = useState<DancePipelineMode>("ltx");

  const currentPresets = PRESET_GROUPS.find((g) => g.key === presetGroup)?.list ?? VIDEO_PRESETS;

  useEffect(() => {
    getHealth()
      .then((r) => setGpuAvailable(r.gpu_available))
      .catch(() => setGpuAvailable(null));

    setDanceListLoading(true);
    Promise.all([getDanceList(), getDanceMotions()])
      .then(([listRes, motions]) => {
        setDanceMotions(motions);
        const dances = listRes.dances ?? [];
        setDanceList(dances);
        if (dances.length > 0) setSelectedMotionId(dances[0]!.id);
        else if (motions.length > 0) setSelectedMotionId(motions[0]!.id);
        else setSelectedMotionId("rat_dance");
      })
      .catch(() => {
        setDanceList([]);
        getDanceMotions().then(setDanceMotions).catch(() => setDanceMotions([]));
      })
      .finally(() => setDanceListLoading(false));
  }, []);

  const handleRefreshDanceList = useCallback(async () => {
    setDanceListRefreshing(true);
    try {
      await refreshDanceList();
      const r = await getDanceList();
      setDanceList(r.dances);
      if (r.dances.length > 0) setSelectedMotionId(r.dances[0]!.id);
    } catch {
      setDanceList([]);
    } finally {
      setDanceListRefreshing(false);
    }
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
      const data = await generateDance(file, selectedMotionId, danceCharacter, dancePipeline);
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
  }, [file, selectedMotionId, danceCharacter, dancePipeline]);

  const videoSrc =
    state.phase === "success"
      ? videoSrcFromApiField(state.data.video_base64, state.data.video_url)
      : "";

  const handleDownload = useCallback(() => {
    if (state.phase !== "success") return;
    const rawB64 = rawBase64FromVideoField(state.data.video_base64);
    if (rawB64) {
      try {
        const bin = atob(rawB64);
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

  const handleCustomDanceGenerate = useCallback(async () => {
    if (!file) {
      setState({ phase: "error", message: "캐릭터 이미지를 업로드해주세요." });
      return;
    }
    if (!customRefVideo) {
      setState({ phase: "error", message: "레퍼런스 동영상을 업로드해주세요." });
      return;
    }
    setState({ phase: "loading" });
    try {
      const data = await generateDanceCustom(file, customRefVideo, customCharacter, customDancePipeline);
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
  }, [file, customRefVideo, customCharacter, customDancePipeline]);

  const isProcessing = state.phase === "loading";
  const canGenerate = file !== null && prompt.trim().length > 0 && !isProcessing;
  const canDanceGenerate = file !== null && selectedMotionId.length > 0 && !isProcessing;
  const canCustomDanceGenerate = file !== null && customRefVideo !== null && !isProcessing;

  return (
    <div className="min-h-screen bg-gray-100 py-8">
      {isProcessing && (
        <LoadingOverlay message="동영상 생성 중... (LTX-2.3 / ComfyUI, 5~15분 소요될 수 있습니다. 완료까지 기다려 주세요)" />
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

          <hr className="border-gray-200" />

          <section>
            <h2 className="mb-1 text-sm font-semibold text-gray-700">사전 등록된 댄스 영상으로 생성</h2>
            <p className="mb-3 text-xs text-gray-500">
              서버에 등록된 댄스 영상 중 하나를 선택하면, 업로드한 캐릭터 이미지가 해당 동작을 따라 움직입니다.
            </p>
            {danceListLoading ? (
              <p className="mb-3 text-sm text-gray-500">댄스 목록 불러오는 중...</p>
            ) : danceList.length > 0 ? (
              <>
                <div className="mb-2 flex items-center justify-between gap-2">
                  <label className="text-xs font-medium text-gray-600">댄스 영상 선택</label>
                  <button
                    type="button"
                    onClick={handleRefreshDanceList}
                    disabled={danceListRefreshing || isProcessing}
                    className="rounded border border-gray-300 bg-white px-2 py-1 text-xs text-gray-600 hover:bg-gray-50 disabled:opacity-50"
                  >
                    {danceListRefreshing ? "새로고침 중..." : "목록 새로고침"}
                  </button>
                </div>
                <div className="mb-3 max-h-48 overflow-y-auto rounded-lg border border-gray-200 bg-gray-50/50 p-2">
                  {danceList.map((d) => (
                    <button
                      key={d.id}
                      type="button"
                      onClick={() => setSelectedMotionId(d.id)}
                      disabled={isProcessing}
                      className={`mb-1 flex w-full items-center justify-between rounded-lg border px-3 py-2 text-left text-sm disabled:opacity-60 ${
                        selectedMotionId === d.id
                          ? "border-indigo-500 bg-indigo-50 text-indigo-800"
                          : "border-gray-200 bg-white text-gray-700 hover:border-indigo-300"
                      }`}
                    >
                      <span className="font-medium">{d.display_name}</span>
                      <span className="text-xs text-gray-500">
                        {d.duration_seconds.toFixed(1)}초 · {d.file_size_mb.toFixed(1)}MB
                      </span>
                    </button>
                  ))}
                </div>
              </>
            ) : (
              <div className="mb-3 rounded-lg border border-amber-200 bg-amber-50/80 p-3 text-sm text-amber-800">
                등록된 댄스 영상이 없습니다. 서버의 motions 폴더에 mp4 파일을 넣고 아래 새로고침을 눌러주세요.
                <button
                  type="button"
                  onClick={handleRefreshDanceList}
                  disabled={danceListRefreshing || isProcessing}
                  className="ml-2 rounded border border-amber-400 bg-white px-2 py-1 text-xs hover:bg-amber-50 disabled:opacity-50"
                >
                  {danceListRefreshing ? "새로고침 중..." : "목록 새로고침"}
                </button>
              </div>
            )}
            <div className="mb-3 flex gap-2">
              {(["dog", "cat"] as const).map((c) => (
                <button
                  key={c}
                  type="button"
                  onClick={() => setDanceCharacter(c)}
                  disabled={isProcessing}
                  className={`rounded-lg border-2 px-3 py-1.5 text-sm font-medium disabled:opacity-60 ${
                    danceCharacter === c
                      ? "border-indigo-500 bg-indigo-50 text-indigo-700"
                      : "border-gray-200 bg-white text-gray-700 hover:border-indigo-400"
                  }`}
                >
                  {c === "dog" ? "강아지" : "고양이"}
                </button>
              ))}
            </div>
            <p className="mb-2 text-xs font-medium text-gray-600">생성 방식 (API)</p>
            <div className="mb-3 flex flex-wrap gap-2">
              <button
                type="button"
                onClick={() => setDancePipeline("ltx")}
                disabled={isProcessing}
                className={`rounded-lg border-2 px-3 py-1.5 text-xs font-medium disabled:opacity-60 ${
                  dancePipeline === "ltx"
                    ? "border-emerald-500 bg-emerald-50 text-emerald-800"
                    : "border-gray-200 bg-white text-gray-700 hover:border-emerald-400"
                }`}
              >
                LTX + 댄스 영상 (기본)
              </button>
              <button
                type="button"
                onClick={() => setDancePipeline("pose_sdxl")}
                disabled={isProcessing}
                className={`rounded-lg border-2 px-3 py-1.5 text-xs font-medium disabled:opacity-60 ${
                  dancePipeline === "pose_sdxl"
                    ? "border-emerald-500 bg-emerald-50 text-emerald-800"
                    : "border-gray-200 bg-white text-gray-700 hover:border-emerald-400"
                }`}
              >
                포즈→ComfyUI 프레임 (pose_sdxl)
              </button>
            </div>
            <p className="mb-3 text-xs text-gray-500">
              pose_sdxl은 서버에 <code className="rounded bg-gray-100 px-1">pipelines/dance/dog_pose_generation.json</code>과 ComfyUI가 필요합니다. 실패 시 자동으로 LTX 경로로 재시도할 수 있습니다.
            </p>
            <p className="mb-3 rounded-md border border-amber-100 bg-amber-50/80 px-2 py-2 text-xs text-amber-900">
              <strong className="font-semibold">동작 방식 안내:</strong> 강아지 사진은 사용자가 올린 이미지를 쓰며, 별도 &quot;강아지 자동 인식&quot; 단계는 없습니다.
              레퍼런스 댄스 영상의 포즈 추출은 <strong className="font-medium">사람 골격(MediaPipe) 기준</strong>이라 네 발 보행과 어긋날 수 있습니다.
              ComfyUI에서 Wan만 단독 실행하는 것과 이 앱의 댄스 API(LTX/pose_sdxl)는 <strong className="font-medium">다른 경로</strong>입니다. 자세한 격차는 백엔드 <code className="rounded bg-white/60 px-0.5">docs/DANCE_GOALS_AND_GAPS.md</code> 참고.
            </p>
            <button
              type="button"
              onClick={handleDanceGenerate}
              disabled={!canDanceGenerate}
              className="w-full rounded-lg bg-emerald-600 px-4 py-3 font-medium text-white hover:bg-emerald-700 disabled:cursor-not-allowed disabled:opacity-50"
            >
              선택한 댄스로 영상 생성
            </button>
          </section>

          <hr className="border-gray-200" />

          <section>
            <h2 className="mb-1 text-sm font-semibold text-gray-700">커스텀 레퍼런스 영상으로 동작 따라하기</h2>
            <p className="mb-3 text-xs text-gray-500">
              원하는 동작이 담긴 영상을 업로드하면 캐릭터 이미지가 해당 동작을 따라합니다.
            </p>
            <div className="mb-3">
              <label className="mb-1 block text-xs font-medium text-gray-600">레퍼런스 동영상 (mp4, mov 등)</label>
              <input
                type="file"
                accept="video/*"
                disabled={isProcessing}
                onChange={(e) => setCustomRefVideo(e.target.files?.[0] ?? null)}
                className="w-full rounded-lg border border-gray-300 px-3 py-2 text-sm file:mr-3 file:rounded file:border-0 file:bg-indigo-50 file:px-3 file:py-1 file:text-xs file:font-medium file:text-indigo-700 disabled:opacity-60"
              />
              {customRefVideo && (
                <p className="mt-1 text-xs text-gray-500">{customRefVideo.name} ({(customRefVideo.size / 1024 / 1024).toFixed(1)}MB)</p>
              )}
            </div>
            <div className="mb-3 flex gap-2">
              {(["dog", "cat"] as const).map((c) => (
                <button
                  key={c}
                  type="button"
                  onClick={() => setCustomCharacter(c)}
                  disabled={isProcessing}
                  className={`rounded-lg border-2 px-3 py-1.5 text-sm font-medium disabled:opacity-60 ${
                    customCharacter === c
                      ? "border-indigo-500 bg-indigo-50 text-indigo-700"
                      : "border-gray-200 bg-white text-gray-700 hover:border-indigo-400"
                  }`}
                >
                  {c === "dog" ? "강아지" : "고양이"}
                </button>
              ))}
            </div>
            <p className="mb-2 text-xs font-medium text-gray-600">생성 방식 (API)</p>
            <div className="mb-3 flex flex-wrap gap-2">
              <button
                type="button"
                onClick={() => setCustomDancePipeline("ltx")}
                disabled={isProcessing}
                className={`rounded-lg border-2 px-3 py-1.5 text-xs font-medium disabled:opacity-60 ${
                  customDancePipeline === "ltx"
                    ? "border-violet-500 bg-violet-50 text-violet-800"
                    : "border-gray-200 bg-white text-gray-700 hover:border-violet-400"
                }`}
              >
                LTX + 레퍼런스 (기본)
              </button>
              <button
                type="button"
                onClick={() => setCustomDancePipeline("pose_sdxl")}
                disabled={isProcessing}
                className={`rounded-lg border-2 px-3 py-1.5 text-xs font-medium disabled:opacity-60 ${
                  customDancePipeline === "pose_sdxl"
                    ? "border-violet-500 bg-violet-50 text-violet-800"
                    : "border-gray-200 bg-white text-gray-700 hover:border-violet-400"
                }`}
              >
                포즈→ComfyUI (pose_sdxl)
              </button>
            </div>
            <p className="mb-3 text-xs text-amber-800/90">
              레퍼런스는 <strong>모션 참고</strong>용입니다. 사람 영상이면 포즈는 인체 기준이며, Wan(ComfyUI 단독)과 앱 API는 별도입니다. → <code className="rounded bg-amber-100/80 px-0.5">backend/docs/DANCE_GOALS_AND_GAPS.md</code>
            </p>
            <button
              type="button"
              onClick={handleCustomDanceGenerate}
              disabled={!canCustomDanceGenerate}
              className="w-full rounded-lg bg-violet-600 px-4 py-3 font-medium text-white hover:bg-violet-700 disabled:cursor-not-allowed disabled:opacity-50"
            >
              레퍼런스 영상으로 동작 생성
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
