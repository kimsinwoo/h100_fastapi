import React from "react";
import { cva, type VariantProps } from "class-variance-authority";
import type { MedicalSeverity } from "shared/medical/types";
import { getSeverityLabel } from "shared/medical/severityTheme";

const severityBadge = cva(
  "inline-flex items-center rounded-full px-2.5 py-0.5 text-xs font-semibold",
  {
    variants: {
      severity: {
        low: "bg-green-100 text-green-800 dark:bg-green-900/30 dark:text-green-300",
        medium: "bg-yellow-100 text-yellow-800 dark:bg-yellow-900/30 dark:text-yellow-300",
        high: "bg-orange-100 text-orange-800 dark:bg-orange-900/30 dark:text-orange-300",
        critical: "bg-red-100 text-red-800 dark:bg-red-900/30 dark:text-red-300",
      },
    },
    defaultVariants: { severity: "medium" },
  }
);

type Props = VariantProps<typeof severityBadge> & {
  severity: MedicalSeverity;
  "aria-label"?: string;
};

export const SeverityBadge = React.memo(function SeverityBadge({
  severity,
  "aria-label": ariaLabel,
}: Props) {
  const label = getSeverityLabel(severity);
  return (
    <span
      className={severityBadge({ severity })}
      role="status"
      aria-label={ariaLabel ?? `심각도: ${label}`}
    >
      {label}
    </span>
  );
});
