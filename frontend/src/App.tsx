import React, { useCallback, useEffect, useState } from "react";
import { ImageUploader } from "./components/ImageUploader";
import { LoadingOverlay } from "./components/LoadingSpinner";
import { ResultViewer } from "./components/ResultViewer";
import { StyleSelector } from "./components/StyleSelector";
import {
  generateImage,
  getHealth,
  getErrorMessage,
  isApiError,
} from "./services/api";
import type { GenerateResponse } from "./types/api";

type AppState =
  | { phase: "idle" }
  | { phase: "loading" }
  | { phase: "success"; data: GenerateResponse }
  | { phase: "error"; message: string };

export default function App() {
  const [file, setFile] = useState<File | null>(null);
  const [style, setStyle] = useState<string>("realistic");
  const [customPrompt, setCustomPrompt] = useState<string>("");
  const [strength, setStrength] = useState<number>(0.6);
  const [gpuAvailable, setGpuAvailable] = useState<boolean | null>(null);
  const [state, setState] = useState<AppState>({ phase: "idle" });

  useEffect(() => {
    getHealth()
      .then((r) => setGpuAvailable(r.gpu_available))
      .catch(() => setGpuAvailable(null));
  }, []);

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
        <header className="mb-8 text-center">
          <h1 className="text-3xl font-bold text-gray-900">Z-Image AI</h1>
          <p className="mt-2 text-gray-600">Transform your images with AI style presets</p>
          {gpuAvailable !== null && (
            <p className="mt-1 text-sm text-gray-500">
              Backend: {gpuAvailable ? "GPU" : "CPU"} mode
            </p>
          )}
        </header>

        <div className="space-y-6 rounded-xl bg-white p-6 shadow">
          <section>
            <h2 className="mb-2 text-sm font-semibold text-gray-700">Upload image</h2>
            <ImageUploader
              onFileSelect={setFile}
              selectedFile={file}
              disabled={isProcessing}
            />
          </section>

          <section>
            <h2 className="mb-2 text-sm font-semibold text-gray-700">Style</h2>
            <StyleSelector
              selected={style}
              onSelect={setStyle}
              disabled={isProcessing}
            />
          </section>

          <section>
            <label htmlFor="custom-prompt" className="mb-2 block text-sm font-semibold text-gray-700">
              Custom prompt (optional)
            </label>
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
            <div
              role="alert"
              className="rounded-lg border border-red-200 bg-red-50 p-4 text-sm text-red-800"
            >
              {state.message}
            </div>
          )}

          {state.phase === "success" && (
            <section className="border-t border-gray-200 pt-6">
              <h2 className="mb-4 text-sm font-semibold text-gray-700">Result</h2>
              <ResultViewer
                originalUrl={state.data.original_url.startsWith("http") ? state.data.original_url : `${window.location.origin}${state.data.original_url}`}
                generatedUrl={state.data.generated_url.startsWith("http") ? state.data.generated_url : `${window.location.origin}${state.data.generated_url}`}
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
