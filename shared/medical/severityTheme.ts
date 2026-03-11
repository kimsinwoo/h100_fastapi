/**
 * Severity-based theme tokens (shared).
 * Used by web (Tailwind/CVA) and native (StyleSheet) to keep semantics identical.
 */

import type { MedicalSeverity } from "./types";

export type SeverityThemeTokens = {
  borderLeft: string;
  badgeBg: string;
  badgeText: string;
  icon: string;
  emergencyBg: string;
  emergencyBorder: string;
  emergencyText: string;
};

const SEVERITY_THEME: Record<MedicalSeverity, SeverityThemeTokens> = {
  low: {
    borderLeft: "border-l-green-500",
    badgeBg: "bg-green-100",
    badgeText: "text-green-800",
    icon: "text-green-600",
    emergencyBg: "bg-red-50",
    emergencyBorder: "border-red-200",
    emergencyText: "text-red-900",
  },
  medium: {
    borderLeft: "border-l-yellow-500",
    badgeBg: "bg-yellow-100",
    badgeText: "text-yellow-800",
    icon: "text-yellow-600",
    emergencyBg: "bg-red-50",
    emergencyBorder: "border-red-200",
    emergencyText: "text-red-900",
  },
  high: {
    borderLeft: "border-l-orange-500",
    badgeBg: "bg-orange-100",
    badgeText: "text-orange-800",
    icon: "text-orange-600",
    emergencyBg: "bg-red-50",
    emergencyBorder: "border-red-200",
    emergencyText: "text-red-900",
  },
  critical: {
    borderLeft: "border-l-red-500",
    badgeBg: "bg-red-100",
    badgeText: "text-red-800",
    icon: "text-red-600",
    emergencyBg: "bg-red-100",
    emergencyBorder: "border-red-300",
    emergencyText: "text-red-900",
  },
};

export function getSeverityTheme(severity: MedicalSeverity): SeverityThemeTokens {
  return SEVERITY_THEME[severity];
}

/** Exhaustive severity label (no fallback). */
export function getSeverityLabel(severity: MedicalSeverity): string {
  switch (severity) {
    case "low":
      return "낮음";
    case "medium":
      return "보통";
    case "high":
      return "높음";
    case "critical":
      return "긴급";
    default: {
      const _: never = severity;
      return _;
    }
  }
}
