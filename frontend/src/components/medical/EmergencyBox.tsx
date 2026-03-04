import React from "react";
import type { EmergencyCriteria } from "shared/medical/types";

interface EmergencyBoxProps {
  emergencyCriteria: EmergencyCriteria;
  animated?: boolean;
  "aria-label"?: string;
}

const EMPTY_ITEMS: string[] = [];

export const EmergencyBox = React.memo(function EmergencyBox({
  emergencyCriteria,
  animated = true,
  "aria-label": ariaLabel,
}: EmergencyBoxProps) {
  const items = emergencyCriteria.items ?? EMPTY_ITEMS;
  if (items.length === 0) return null;

  return (
    <div
      className={`rounded-lg border-2 border-red-200 bg-red-50 p-3 transition-opacity duration-300 dark:border-red-900 dark:bg-red-950/50 ${
        animated ? "opacity-100" : "opacity-100"
      }`}
      role="alert"
      aria-label={ariaLabel ?? "즉시 병원 내원 기준"}
    >
      <p className="mb-2 text-sm font-semibold text-red-900 dark:text-red-200">
        즉시 병원 내원 기준
      </p>
      <ul className="list-inside list-disc space-y-1 text-sm text-red-800 dark:text-red-300">
        {items.map((item, i) => (
          <li key={i}>{item}</li>
        ))}
      </ul>
    </div>
  );
});
