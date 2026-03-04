import React from "react";

interface SummaryBlockProps {
  summary: string;
  "aria-label"?: string;
}

export const SummaryBlock = React.memo(function SummaryBlock({
  summary,
  "aria-label": ariaLabel,
}: SummaryBlockProps) {
  return (
    <div
      className="rounded-lg border border-gray-200 bg-gray-50/50 p-3 text-sm text-gray-800 dark:border-gray-700 dark:bg-gray-800/50 dark:text-gray-200"
      aria-label={ariaLabel ?? "요약"}
    >
      <p className="whitespace-pre-wrap leading-relaxed">{summary}</p>
    </div>
  );
});
