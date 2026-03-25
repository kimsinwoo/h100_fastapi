import { useCallback, useEffect, useState } from "react";
import { Link } from "react-router-dom";
import { ImageUploader } from "../components/ImageUploader";
import { LoadingOverlay } from "../components/LoadingSpinner";
import { ResultViewer } from "../components/ResultViewer";
import { StyleSelector } from "../components/StyleSelector";
import {
  generateImage,
  getHealth,
  getStyles,
  getLlmStatus,
  getErrorMessage,
  suggestPrompt,
  getApiResourceUrl,
  acAnalyze,
  acReconstruct,
  analyzeUniversal,
} from "../services/api";
import type {
  GenerateResponse,
  StylesResponse,
  ACBiologicalAnalysis,
  UniversalAnalysisResponse,
} from "../types/api";

type AppState =
  | { phase: "idle" }
  | { phase: "loading" }
  | { phase: "success"; data: GenerateResponse }
  | { phase: "error"; message: string };

type ACModeState =
  | { step: "idle" }
  | { step: "analyzed"; data: ACBiologicalAnalysis }
  | { step: "loading" }
  | { step: "success"; data: GenerateResponse }
  | { step: "error"; message: string };

const AC_SPECIES_OPTIONS = [
  { value: "cat", label: "고양이" },
  { value: "dog", label: "강아지" },
  { value: "rabbit", label: "토끼" },
  { value: "hamster", label: "햄스터" },
  { value: "bird", label: "새" },
  { value: "other", label: "기타" },
] as const;

export default function GeneratePage() {
  const [tab, setTab] = useState<"style" | "ac">("style");
  const [file, setFile] = useState<File | null>(null);
  const [styles, setStyles] = useState<StylesResponse | null>(null);
  const [style, setStyle] = useState<string>("sailor_moon");
  const [species, setSpecies] = useState<"dog" | "cat">("dog");
  const [customPrompt, setCustomPrompt] = useState<string>("");
  const [strength, setStrength] = useState<number>(0.5);
  const [gpuAvailable, setGpuAvailable] = useState<boolean | null>(null);
  const [llmAvailable, setLlmAvailable] = useState<boolean>(false);
  const [llmModel, setLlmModel] = useState<string | null>(null);
  const [suggesting, setSuggesting] = useState(false);
  /** Universal 분석 + 포즈 락·드리프트 시 1회 재생성 (체감 품질↑, 시간↑). */
  const [qualityBoost, setQualityBoost] = useState(true);
  const [state, setState] = useState<AppState>({ phase: "idle" });

  // AC 주민 재구성: 폼 + 2단계 상태
  const [acFile, setAcFile] = useState<File | null>(null);
  const [acSpecies, setAcSpecies] = useState<string>("cat");
  const [acMainColor, setAcMainColor] = useState<string>("cream");
  const [acSecondaryColor, setAcSecondaryColor] = useState<string>("none");
  const [acEyeColor, setAcEyeColor] = useState<string>("amber");
  const [acMarkings, setAcMarkings] = useState<string>("none");
  const [acState, setAcState] = useState<ACModeState>({ step: "idle" });

  useEffect(() => {
    getHealth()
      .then((r) => setGpuAvailable(r.gpu_available))
      .catch(() => setGpuAvailable(null));
    getLlmStatus()
      .then((r) => {
        setLlmAvailable(r.available);
        setLlmModel(r.model ?? null);
      })
      .catch(() => setLlmAvailable(false));
    getStyles()
      .then((s) => {
        setStyles(s);
        return s;
      })
      .catch(() => setStyles({}));
  }, []);

  // 스타일 목록 로드 후 현재 선택이 목록에 없으면 첫 번째 스타일로
  useEffect(() => {
    const keys = styles ? Object.keys(styles) : [];
    if (keys.length === 0) return;
    if (!Object.prototype.hasOwnProperty.call(styles!, style)) {
      setStyle(keys[0]!);
    }
  }, [styles, style]);

  const handleSuggestPrompt = useCallback(async () => {
    setSuggesting(true);
    try {
      const hint = customPrompt.trim() || undefined;
      const prompt = await suggestPrompt(style, hint);
      setCustomPrompt(prompt);
    } catch {
      setState({ phase: "error", message: "LLM 프롬프트 추천 실패" });
    } finally {
      setSuggesting(false);
    }
  }, [style, customPrompt]);

  const handleGenerate = useCallback(async () => {
    if (!file) return;
    setState({ phase: "loading" });
    try {
      const baseOpts = { species, validateAndRetry: false as boolean };
      let opts:
        | typeof baseOpts
        | (typeof baseOpts & {
            usePoseLock: true;
            validateAndRetry: true;
            analysis: UniversalAnalysisResponse;
          }) = baseOpts;

      if (qualityBoost) {
        try {
          const analysis = await analyzeUniversal({ file, species });
          opts = {
            species,
            usePoseLock: true,
            validateAndRetry: true,
            analysis,
          };
        } catch {
          /* 분석 실패 시 일반 파이프라인만 사용 */
        }
      }

      const data = await generateImage(file, style, customPrompt || null, strength, null, undefined, opts);
      setState({ phase: "success", data });
    } catch (err) {
      const message = getErrorMessage(err);
      setState({ phase: "error", message });
    }
  }, [file, style, species, customPrompt, strength, qualityBoost]);

  const handleDownload = useCallback(() => {
    if (state.phase !== "success") return;
    const b64 = state.data.generated_image_base64;
    if (b64) {
      try {
        const bin = atob(b64);
        const bytes = new Uint8Array(bin.length);
        for (let i = 0; i < bin.length; i++) bytes[i] = bin.charCodeAt(i);
        const blob = new Blob([bytes], { type: "image/png" });
        const a = document.createElement("a");
        a.href = URL.createObjectURL(blob);
        a.download = `generated-${Date.now()}.png`;
        a.click();
        URL.revokeObjectURL(a.href);
      } catch {
        setState({ phase: "error", message: "Download failed" });
      }
      return;
    }
    const fullUrl = getApiResourceUrl(state.data.generated_url);
    fetch(fullUrl)
      .then((r) => r.blob())
      .then((blob) => {
        const a = document.createElement("a");
        a.href = URL.createObjectURL(blob);
        a.download = `generated-${Date.now()}.png`;
        a.click();
        URL.revokeObjectURL(a.href);
      })
      .catch(() => setState({ phase: "error", message: "Download failed" }));
  }, [state]);

  // AC 주민 재구성: 1단계 분석
  const handleAcAnalyze = useCallback(async () => {
    setAcState({ step: "loading" });
    try {
      const data = await acAnalyze({
        file: acFile ?? undefined,
        species: acSpecies,
        main_fur_color: acMainColor,
        secondary_fur_color: acSecondaryColor,
        eye_color: acEyeColor,
        markings: acMarkings,
      });
      setAcState({ step: "analyzed", data });
    } catch (err) {
      setAcState({ step: "error", message: getErrorMessage(err) });
    }
  }, [acFile, acSpecies, acMainColor, acSecondaryColor, acEyeColor, acMarkings]);

  // AC 주민 재구성: 2단계 생성 (분석 결과 또는 폼 값 사용)
  const handleAcReconstruct = useCallback(async () => {
    const payload =
      acState.step === "analyzed"
        ? {
            species: acState.data.species,
            main_fur_color: acState.data.main_fur_color,
            secondary_fur_color: acState.data.secondary_fur_color,
            eye_color: acState.data.eye_color,
            markings: acState.data.markings,
            ear_type: acState.data.ear_type,
            tail_type: acState.data.tail_type,
          }
        : {
            species: acSpecies,
            main_fur_color: acMainColor,
            secondary_fur_color: acSecondaryColor,
            eye_color: acEyeColor,
            markings: acMarkings,
            ear_type: null as string | null,
            tail_type: null as string | null,
          };
    setAcState({ step: "loading" });
    try {
      const result = await acReconstruct(payload);
      setAcState({ step: "success", data: result });
    } catch (err) {
      setAcState({ step: "error", message: getErrorMessage(err) });
    }
  }, [acState, acSpecies, acMainColor, acSecondaryColor, acEyeColor, acMarkings]);

  const handleAcDownload = useCallback(() => {
    if (acState.step !== "success") return;
    const d = acState.data;
    const b64 = d.generated_image_base64;
    if (b64) {
      try {
        const bin = atob(b64);
        const bytes = new Uint8Array(bin.length);
        for (let i = 0; i < bin.length; i++) bytes[i] = bin.charCodeAt(i);
        const blob = new Blob([bytes], { type: "image/png" });
        const a = document.createElement("a");
        a.href = URL.createObjectURL(blob);
        a.download = `ac-villager-${Date.now()}.png`;
        a.click();
        URL.revokeObjectURL(a.href);
      } catch {
        setAcState({ step: "error", message: "Download failed" });
      }
      return;
    }
    const fullUrl = getApiResourceUrl(d.generated_url);
    fetch(fullUrl)
      .then((r) => r.blob())
      .then((blob) => {
        const a = document.createElement("a");
        a.href = URL.createObjectURL(blob);
        a.download = `ac-villager-${Date.now()}.png`;
        a.click();
        URL.revokeObjectURL(a.href);
      })
      .catch(() => setAcState({ step: "error", message: "Download failed" }));
  }, [acState]);

  const isProcessing = state.phase === "loading";
  const canGenerate = file !== null && !isProcessing;
  const acLoading = acState.step === "loading";
  const acCanReconstruct =
    !acLoading &&
    (acState.step === "analyzed" ||
      (acState.step === "idle" && acSpecies) ||
      (acState.step === "error" && acSpecies));

  return (
    <div className="min-h-screen bg-gray-100 py-8">
      {(isProcessing || acLoading) && (
        <LoadingOverlay
          message={
            acLoading
              ? "게임 캐릭터 생성 중..."
              : qualityBoost && tab === "style"
                ? "이미지 생성 중... (포즈 분석·생성으로 2~4분 걸릴 수 있습니다)"
                : "이미지 생성 중... (1~2분 소요될 수 있습니다)"
          }
        />
      )}

      <div className="mx-auto max-w-4xl px-4">
        <header className="mb-8 flex items-center justify-between">
          <div className="text-center">
            <h1 className="text-3xl font-bold text-gray-900">Z-Image AI</h1>
            <p className="mt-2 text-gray-600">Transform your images with AI style presets</p>
            <div className="mt-1 flex flex-wrap justify-center gap-3 text-sm text-gray-500">
              {gpuAvailable !== null && <span>Backend: {gpuAvailable ? "GPU" : "CPU"}</span>}
              {llmAvailable && llmModel && <span>LLM: {llmModel}</span>}
              {!llmAvailable && gpuAvailable !== null && <span>LLM: 사용 불가</span>}
            </div>
          </div>
          <div className="flex gap-2">
            <Link
              to="/video"
              className="rounded-lg border border-gray-300 bg-white px-4 py-2 text-sm font-medium text-gray-700 hover:bg-gray-50"
            >
              이미지→동영상
            </Link>
            <Link
              to="/chat"
              className="rounded-lg border border-gray-300 bg-white px-4 py-2 text-sm font-medium text-gray-700 hover:bg-gray-50"
            >
              LLM 채팅
            </Link>
            <Link
              to="/training"
              className="rounded-lg border border-gray-300 bg-white px-4 py-2 text-sm font-medium text-gray-700 hover:bg-gray-50"
            >
              LoRA 학습
            </Link>
          </div>
        </header>

        {/* 탭: 일반 스타일 | 게임 캐릭터 재구성 */}
        <div className="mb-4 flex gap-2">
          <button
            type="button"
            onClick={() => setTab("style")}
            className={`rounded-lg border px-4 py-2 text-sm font-medium ${
              tab === "style"
                ? "border-indigo-600 bg-indigo-600 text-white"
                : "border-gray-300 bg-white text-gray-700 hover:bg-gray-50"
            }`}
          >
            일반 스타일 변환
          </button>
          <button
            type="button"
            onClick={() => setTab("ac")}
            className={`rounded-lg border px-4 py-2 text-sm font-medium ${
              tab === "ac"
                ? "border-indigo-600 bg-indigo-600 text-white"
                : "border-gray-300 bg-white text-gray-700 hover:bg-gray-50"
            }`}
          >
            게임 캐릭터 재구성 (2단계)
          </button>
        </div>

        {tab === "style" && (
        <div className="space-y-6 rounded-xl bg-white p-6 shadow">
          <section>
            <h2 className="mb-2 text-sm font-semibold text-gray-700">1. 이미지 업로드</h2>
            <ImageUploader onFileSelect={setFile} selectedFile={file} disabled={isProcessing} />
          </section>
          <section>
            <h2 className="mb-2 text-sm font-semibold text-gray-700">2. 반려동물 종류</h2>
            <p className="mb-2 text-xs text-gray-500">선택한 종에 맞는 프롬프트가 적용됩니다.</p>
            <div className="flex gap-2">
              <button
                type="button"
                onClick={() => setSpecies("dog")}
                disabled={isProcessing}
                className={`rounded-lg border-2 px-4 py-2 text-sm font-medium disabled:opacity-60 ${
                  species === "dog"
                    ? "border-indigo-600 bg-indigo-50 text-indigo-700"
                    : "border-gray-200 bg-white text-gray-700 hover:border-indigo-300"
                }`}
              >
                강아지
              </button>
              <button
                type="button"
                onClick={() => setSpecies("cat")}
                disabled={isProcessing}
                className={`rounded-lg border-2 px-4 py-2 text-sm font-medium disabled:opacity-60 ${
                  species === "cat"
                    ? "border-indigo-600 bg-indigo-50 text-indigo-700"
                    : "border-gray-200 bg-white text-gray-700 hover:border-indigo-300"
                }`}
              >
                고양이
              </button>
            </div>
          </section>
          <section>
            <div className="mb-2 flex items-center justify-between">
              <label htmlFor="custom-prompt" className="text-sm font-semibold text-gray-700">
                3. 편집 지시 (직접 입력)
              </label>
              {llmAvailable && (
                <button
                  type="button"
                  onClick={handleSuggestPrompt}
                  disabled={suggesting || isProcessing}
                  className="rounded bg-emerald-600 px-2 py-1 text-xs font-medium text-white hover:bg-emerald-700 disabled:opacity-50"
                >
                  {suggesting ? "작성중...🤔" : "AI 프롬프트 추천"}
                </button>
              )}
            </div>
            <textarea
              id="custom-prompt"
              value={customPrompt}
              onChange={(e) => setCustomPrompt(e.target.value)}
              disabled={isProcessing}
              placeholder="예: pixel art, 8-bit — 구체적으로 적을수록 좋습니다. 비워도 OmniGen 기본 스타일 지시가 적용됩니다."
              rows={3}
              className="w-full rounded-lg border border-gray-300 px-3 py-2 text-sm focus:border-indigo-500 focus:outline-none focus:ring-1 focus:ring-indigo-500 disabled:opacity-60"
            />
          </section>
          <section>
            <h2 className="mb-2 text-sm font-semibold text-gray-700">4. Style (2D 캐릭터 재해석)</h2>
            <StyleSelector styles={styles} selected={style} onSelect={setStyle} disabled={isProcessing} />
          </section>
          <section className="rounded-lg border border-gray-100 bg-gray-50/80 p-3">
            <label className="flex cursor-pointer items-start gap-2 text-sm text-gray-800">
              <input
                type="checkbox"
                checked={qualityBoost}
                onChange={(e) => setQualityBoost(e.target.checked)}
                disabled={isProcessing}
                className="mt-0.5 accent-indigo-600"
              />
              <span>
                <span className="font-semibold">품질 보강 (권장)</span>
                <span className="block text-xs font-normal text-gray-600">
                  자세·의류 등을 분석한 뒤 포즈를 맞추고, 결과가 어긋나면 1회 재시도합니다. 첫 실행에 분석 API가 한 번 더 호출됩니다.
                </span>
              </span>
            </label>
          </section>
          <section>
            <label htmlFor="strength" className="mb-2 block text-sm font-semibold text-gray-700">
              Strength: {strength.toFixed(2)}
            </label>
            <input
              id="strength"
              type="range"
              min={0.2}
              max={1}
              step={0.05}
              value={strength}
              onChange={(e) => setStrength(Number(e.target.value))}
              disabled={isProcessing}
              className="w-full accent-indigo-600 disabled:opacity-60"
            />
          </section>
          <div>
            <button
              type="button"
              onClick={handleGenerate}
              disabled={!canGenerate}
              className="w-full rounded-lg bg-indigo-600 px-4 py-3 font-medium text-white hover:bg-indigo-700 disabled:cursor-not-allowed disabled:opacity-50"
            >
              Generate
            </button>
          </div>
          {state.phase === "error" && (
            <div role="alert" className="rounded-lg border border-red-200 bg-red-50 p-4 text-sm text-red-800">
              {state.message}
            </div>
          )}
          {state.phase === "success" && (
            <section className="border-t border-gray-200 pt-6">
              <h2 className="mb-4 text-sm font-semibold text-gray-700">Result</h2>
              <ResultViewer
                originalUrl={getApiResourceUrl(state.data.original_url)}
                generatedUrl={getApiResourceUrl(state.data.generated_url)}
                generatedImageBase64={state.data.generated_image_base64}
                processingTime={state.data.processing_time}
                onDownload={handleDownload}
              />
            </section>
          )}
        </div>
        )}

        {tab === "ac" && (
          <div className="space-y-6 rounded-xl bg-white p-6 shadow">
            <p className="text-sm text-gray-600">
              업로드 이미지는 참고용입니다. 1단계에서 종·색·무늬를 확인한 뒤, 2단계에서 텍스트만으로 게임 캐릭터 비율로 생성합니다.
            </p>
            <section>
              <h2 className="mb-2 text-sm font-semibold text-gray-700">1단계: 참고 이미지 (선택) + 생물학적 정보</h2>
              <ImageUploader onFileSelect={setAcFile} selectedFile={acFile} disabled={acLoading} />
              <div className="mt-4 grid gap-3 sm:grid-cols-2">
                <div>
                  <label className="mb-1 block text-xs font-medium text-gray-600">종</label>
                  <select
                    value={acSpecies}
                    onChange={(e) => setAcSpecies(e.target.value)}
                    disabled={acLoading}
                    className="w-full rounded border border-gray-300 px-3 py-2 text-sm"
                  >
                    {AC_SPECIES_OPTIONS.map((o) => (
                      <option key={o.value} value={o.value}>{o.label}</option>
                    ))}
                  </select>
                </div>
                <div>
                  <label className="mb-1 block text-xs font-medium text-gray-600">주요 털색</label>
                  <input
                    type="text"
                    value={acMainColor}
                    onChange={(e) => setAcMainColor(e.target.value)}
                    disabled={acLoading}
                    placeholder="cream, orange, gray..."
                    className="w-full rounded border border-gray-300 px-3 py-2 text-sm"
                  />
                </div>
                <div>
                  <label className="mb-1 block text-xs font-medium text-gray-600">보조 털색</label>
                  <input
                    type="text"
                    value={acSecondaryColor}
                    onChange={(e) => setAcSecondaryColor(e.target.value)}
                    disabled={acLoading}
                    placeholder="none, white..."
                    className="w-full rounded border border-gray-300 px-3 py-2 text-sm"
                  />
                </div>
                <div>
                  <label className="mb-1 block text-xs font-medium text-gray-600">눈 색</label>
                  <input
                    type="text"
                    value={acEyeColor}
                    onChange={(e) => setAcEyeColor(e.target.value)}
                    disabled={acLoading}
                    placeholder="amber, green, blue..."
                    className="w-full rounded border border-gray-300 px-3 py-2 text-sm"
                  />
                </div>
                <div className="sm:col-span-2">
                  <label className="mb-1 block text-xs font-medium text-gray-600">주요 무늬</label>
                  <input
                    type="text"
                    value={acMarkings}
                    onChange={(e) => setAcMarkings(e.target.value)}
                    disabled={acLoading}
                    placeholder="none, white chest, stripes..."
                    className="w-full rounded border border-gray-300 px-3 py-2 text-sm"
                  />
                </div>
              </div>
              <div className="mt-3 flex gap-2">
                <button
                  type="button"
                  onClick={handleAcAnalyze}
                  disabled={acLoading}
                  className="rounded-lg bg-slate-600 px-4 py-2 text-sm font-medium text-white hover:bg-slate-700 disabled:opacity-50"
                >
                  1단계: 분석
                </button>
                {acState.step === "analyzed" && (
                  <span className="flex items-center text-sm text-green-700">분석 완료 ✓</span>
                )}
              </div>
            </section>
            {acState.step === "analyzed" && (
              <section className="rounded-lg border border-gray-200 bg-gray-50 p-4 text-sm">
                <h3 className="mb-2 font-semibold text-gray-700">분석 결과</h3>
                <ul className="space-y-1 text-gray-600">
                  <li>종: {acState.data.species}</li>
                  <li>주요 털색: {acState.data.main_fur_color}, 보조: {acState.data.secondary_fur_color}</li>
                  <li>눈: {acState.data.eye_color}, 무늬: {acState.data.markings}</li>
                  <li>귀: {acState.data.ear_type}, 꼬리: {acState.data.tail_type}</li>
                </ul>
              </section>
            )}
            <section>
              <button
                type="button"
                onClick={handleAcReconstruct}
                disabled={!acCanReconstruct}
                className="w-full rounded-lg bg-indigo-600 px-4 py-3 font-medium text-white hover:bg-indigo-700 disabled:cursor-not-allowed disabled:opacity-50"
              >
                2단계: 게임 캐릭터 생성 (T2I 전용)
              </button>
            </section>
            {acState.step === "error" && (
              <div role="alert" className="rounded-lg border border-red-200 bg-red-50 p-4 text-sm text-red-800">
                {acState.message}
              </div>
            )}
            {acState.step === "success" && (
              <section className="border-t border-gray-200 pt-6">
                <h2 className="mb-4 text-sm font-semibold text-gray-700">결과</h2>
                <ResultViewer
                  originalUrl=""
                  generatedUrl={getApiResourceUrl(acState.data.generated_url)}
                  generatedImageBase64={acState.data.generated_image_base64}
                  processingTime={acState.data.processing_time}
                  onDownload={handleAcDownload}
                />
              </section>
            )}
          </div>
        )}
      </div>
    </div>
  );
}
