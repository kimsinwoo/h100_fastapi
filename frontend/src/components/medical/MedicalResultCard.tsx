import React, { useCallback } from "react";
import type { MedicalChatStructured } from "shared/medical/types";
import { getSeverityTheme } from "shared/medical/severityTheme";
import { useMedicalResultStore } from "./medicalResultStore";
import { SeverityBadge } from "./SeverityBadge";
import { SummaryBlock } from "./SummaryBlock";
import { ExplanationAccordion } from "./ExplanationAccordion";
import { EmergencyBox } from "./EmergencyBox";
import { FollowUpQuestionList } from "./FollowUpQuestionList";
import { ActionButtonGroup } from "./ActionButtonGroup";

interface MedicalResultCardProps {
  data: MedicalChatStructured;
  onPrimaryAction?: () => void;
  onSecondaryAction?: () => void;
  actionDisabled?: boolean;
  className?: string;
  "aria-label"?: string;
}

const EMPTY_EXPLANATIONS: MedicalChatStructured["explanations"] = [];
const EMPTY_QUESTIONS: MedicalChatStructured["followUpQuestions"] = [];

export const MedicalResultCard = React.memo(function MedicalResultCard({
  data,
  onPrimaryAction,
  onSecondaryAction,
  actionDisabled = false,
  className = "",
  "aria-label": ariaLabel,
}: MedicalResultCardProps) {
  const theme = getSeverityTheme(data.severity);
  const expandedIds = useMedicalResultStore((s) => s.expandedExplanationIds);
  const copiedQuestionId = useMedicalResultStore((s) => s.copiedQuestionId);
  const toggleExplanation = useMedicalResultStore((s) => s.toggleExplanation);
  const setCopiedQuestionId = useMedicalResultStore((s) => s.setCopiedQuestionId);

  const handleCopy = useCallback((id: string) => {
    setCopiedQuestionId(id);
    setTimeout(() => setCopiedQuestionId(null), 2000);
  }, [setCopiedQuestionId]);

  const explanations = data.explanations ?? EMPTY_EXPLANATIONS;
  const followUpQuestions = data.followUpQuestions ?? EMPTY_QUESTIONS;

  const isCritical = data.severity === "critical";

  return (
    <>
      <article
        className={`overflow-hidden rounded-xl border border-gray-200 bg-white shadow-sm dark:border-gray-700 dark:bg-gray-800 ${theme.borderLeft} border-l-4 ${className}`}
        aria-label={ariaLabel ?? "의료 AI 응답"}
      >
        <div className="p-4 space-y-4">
        <div className="flex items-center justify-between gap-2">
          <SeverityBadge severity={data.severity} />
        </div>

        <SummaryBlock summary={data.summary} />

        {explanations.length > 0 && (
          <ExplanationAccordion
            explanations={explanations}
            expandedIds={expandedIds}
            onToggle={toggleExplanation}
          />
        )}

        {data.emergencyCriteria.items.length > 0 && (
          <EmergencyBox emergencyCriteria={data.emergencyCriteria} animated />
        )}

        {followUpQuestions.length > 0 && (
          <FollowUpQuestionList
            questions={followUpQuestions}
            copiedId={copiedQuestionId}
            onCopy={handleCopy}
          />
        )}

        <ActionButtonGroup
          actions={data.recommendedActions}
          onPrimary={onPrimaryAction ?? (() => {})}
          onSecondary={onSecondaryAction}
          disabled={actionDisabled}
        />
        </div>
      </article>
      {isCritical && data.emergencyCriteria.items.length > 0 && (
        <div
          className="sticky bottom-0 left-0 right-0 z-10 mt-4 flex justify-center rounded-lg border-2 border-red-300 bg-red-100 p-3 dark:border-red-700 dark:bg-red-900/50"
          role="alert"
        >
          <button
            type="button"
            onClick={onPrimaryAction}
            disabled={actionDisabled}
            className="rounded-lg bg-red-600 px-4 py-2 text-sm font-semibold text-white hover:bg-red-700 disabled:opacity-50"
            aria-label="긴급 권장 행동"
          >
            {data.recommendedActions.primary}
          </button>
        </div>
      )}
    </>
  );
});
