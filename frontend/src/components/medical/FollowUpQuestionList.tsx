import React, { useCallback } from "react";
import type { FollowUpQuestion } from "shared/medical/types";

interface FollowUpQuestionListProps {
  questions: FollowUpQuestion[];
  copiedId: string | null;
  onCopy: (id: string) => void;
  "aria-label"?: string;
}

export const FollowUpQuestionList = React.memo(function FollowUpQuestionList({
  questions,
  copiedId,
  onCopy,
  "aria-label": ariaLabel,
}: FollowUpQuestionListProps) {
  if (questions.length === 0) return null;

  return (
    <div
      className="space-y-2"
      role="region"
      aria-label={ariaLabel ?? "이어서 물어보기"}
    >
      <p className="text-xs font-semibold text-gray-600 dark:text-gray-400">
        이어서 물어보기
      </p>
      <ul className="flex flex-wrap gap-2">
        {questions.map((q) => (
          <FollowUpQuestionChip
            key={q.id}
            question={q}
            isCopied={copiedId === q.id}
            onCopy={onCopy}
          />
        ))}
      </ul>
    </div>
  );
});

interface FollowUpQuestionChipProps {
  question: FollowUpQuestion;
  isCopied: boolean;
  onCopy: (id: string) => void;
}

const FollowUpQuestionChip = React.memo(function FollowUpQuestionChip({
  question,
  isCopied,
  onCopy,
}: FollowUpQuestionChipProps) {
  const handleClick = useCallback(() => {
    onCopy(question.id);
    if (typeof navigator?.clipboard?.writeText === "function") {
      navigator.clipboard.writeText(question.question);
    }
  }, [question.id, question.question, onCopy]);

  return (
    <li>
      <button
        type="button"
        onClick={handleClick}
        className="rounded-lg border border-indigo-200 bg-indigo-50 px-3 py-2 text-xs font-medium text-indigo-700 transition hover:bg-indigo-100 focus:outline-none focus:ring-2 focus:ring-indigo-500 dark:border-indigo-800 dark:bg-indigo-900/30 dark:text-indigo-300 dark:hover:bg-indigo-900/50"
        aria-label={`질문 복사: ${question.question}`}
      >
        {isCopied ? "복사됨" : question.question}
      </button>
    </li>
  );
});
