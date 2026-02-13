import React from "react";

interface ResultViewerProps {
  originalUrl: string;
  generatedUrl: string;
  processingTime: number;
  onDownload: () => void;
}

export const ResultViewer: React.FC<ResultViewerProps> = ({
  originalUrl,
  generatedUrl,
  processingTime,
  onDownload,
}) => (
  <div className="space-y-4">
    <div className="grid grid-cols-1 gap-4 md:grid-cols-2">
      <div>
        <p className="mb-2 text-sm font-medium text-gray-600">Original</p>
        <img
          src={originalUrl}
          alt="Original"
          className="max-h-96 w-full rounded-lg border border-gray-200 object-contain"
        />
      </div>
      <div>
        <p className="mb-2 text-sm font-medium text-gray-600">Generated</p>
        <img
          src={generatedUrl}
          alt="Generated"
          className="max-h-96 w-full rounded-lg border border-gray-200 object-contain"
        />
      </div>
    </div>
    <div className="flex flex-wrap items-center gap-4">
      <span className="text-sm text-gray-500">
        Processing time: <strong>{processingTime.toFixed(2)}s</strong>
      </span>
      <button
        type="button"
        onClick={onDownload}
        className="rounded-lg bg-indigo-600 px-4 py-2 text-sm font-medium text-white hover:bg-indigo-700"
      >
        Download generated image
      </button>
    </div>
  </div>
);
