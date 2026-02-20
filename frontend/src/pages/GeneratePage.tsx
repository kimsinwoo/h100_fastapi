import { useCallback, useEffect, useState } from "react";
import { Link } from "react-router-dom";
import { ImageUploader } from "../components/ImageUploader";
import { LoadingOverlay } from "../components/LoadingSpinner";
import { ResultViewer } from "../components/ResultViewer";
import { StyleSelector } from "../components/StyleSelector";
import { generateImage, getHealth, getLlmStatus, getErrorMessage, suggestPrompt } from "../services/api";
import type { GenerateResponse } from "../types/api";

type AppState =
  | { phase: "idle" }
  | { phase: "loading" }
  | { phase: "success"; data: GenerateResponse }
  | { phase: "error"; message: string };

export default function GeneratePage() {
  const [file, setFile] = useState<File | null>(null);
  const [style, setStyle] = useState<string>("realistic");
  const [customPrompt, setCustomPrompt] = useState<string>("");
  const [strength, setStrength] = useState<number>(0.6);
  const [gpuAvailable, setGpuAvailable] = useState<boolean | null>(null);
  const [llmAvailable, setLlmAvailable] = useState<boolean>(false);
  const [llmModel, setLlmModel] = useState<string | null>(null);
  const [suggesting, setSuggesting] = useState(false);
  const [state, setState] = useState<AppState>({ phase: "idle" });

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
  }, []);

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
      const data = await generateImage(file, style, customPrompt || null, strength, null);
      setState({ phase: "success", data });
    } catch (err) {
      const message = getErrorMessage(err);
      setState({ phase: "error", message });
    }
  }, [file, style, customPrompt, strength]);

  const handleDownload = useCallback(() => {
    if (state.phase !== "success") return;
    const url = state.data.generated_url;
    const fullUrl = url.startsWith("http") ? url : `${window.location.origin}${url}`;
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

  const isProcessing = state.phase === "loading";
  const canGenerate = file !== null && !isProcessing;

  return (
    <div className="min-h-screen bg-gray-100 py-8">
      {isProcessing && <LoadingOverlay message="Generating image..." />}

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

        <div className="space-y-6 rounded-xl bg-white p-6 shadow">
          <section>
            <h2 className="mb-2 text-sm font-semibold text-gray-700">Upload image</h2>
            <ImageUploader onFileSelect={setFile} selectedFile={file} disabled={isProcessing} />
          </section>
          <section>
            <h2 className="mb-2 text-sm font-semibold text-gray-700">Style</h2>
            <StyleSelector selected={style} onSelect={setStyle} disabled={isProcessing} />
          </section>
          <section>
            <div className="mb-2 flex items-center justify-between">
              <label htmlFor="custom-prompt" className="text-sm font-semibold text-gray-700">
                Custom prompt (optional)
              </label>
              {llmAvailable && (
                <button
                  type="button"
                  onClick={handleSuggestPrompt}
                  disabled={suggesting || isProcessing}
                  className="rounded bg-emerald-600 px-2 py-1 text-xs font-medium text-white hover:bg-emerald-700 disabled:opacity-50"
                >
                  {suggesting ? "추천 중…" : "AI 프롬프트 추천"}
                </button>
              )}
            </div>
            <textarea
              id="custom-prompt"
              value={customPrompt}
              onChange={(e) => setCustomPrompt(e.target.value)}
              disabled={isProcessing}
              placeholder="e.g. sunset, mountains..."
              rows={2}
              className="w-full rounded-lg border border-gray-300 px-3 py-2 text-sm focus:border-indigo-500 focus:outline-none focus:ring-1 focus:ring-indigo-500 disabled:opacity-60"
            />
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
                originalUrl={
                  state.data.original_url.startsWith("http")
                    ? state.data.original_url
                    : `${window.location.origin}${state.data.original_url}`
                }
                generatedUrl={
                  state.data.generated_url.startsWith("http")
                    ? state.data.generated_url
                    : `${window.location.origin}${state.data.generated_url}`
                }
                processingTime={state.data.processing_time}
                onDownload={handleDownload}
              />
            </section>
          )}
        </div>
      </div>
    </div>
  );
}
