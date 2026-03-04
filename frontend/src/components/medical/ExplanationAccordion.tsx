import React, { useRef, useEffect, useState } from "react";
import type { MedicalExplanationSection } from "shared/medical/types";
import { ExplanationItem } from "./ExplanationItem";

const COLLAPSED = "0fr";
const EXPANDED = "1fr";

interface ExplanationAccordionProps {
  explanations: MedicalExplanationSection[];
  expandedIds: Set<string>;
  onToggle: (id: string) => void;
  "aria-label"?: string;
}

export const ExplanationAccordion = React.memo(function ExplanationAccordion({
  explanations,
  expandedIds,
  onToggle,
  "aria-label": ariaLabel,
}: ExplanationAccordionProps) {
  if (explanations.length === 0) return null;

  return (
    <div
      className="space-y-1"
      role="region"
      aria-label={ariaLabel ?? "의학적 설명"}
    >
      {explanations.map((section) => (
        <AccordionItem
          key={section.id}
          section={section}
          isExpanded={expandedIds.has(section.id)}
          onToggle={onToggle}
        />
      ))}
    </div>
  );
});

interface AccordionItemProps {
  section: MedicalExplanationSection;
  isExpanded: boolean;
  onToggle: (id: string) => void;
}

const AccordionItem = React.memo(function AccordionItem({
  section,
  isExpanded,
  onToggle,
}: AccordionItemProps) {
  const [height, setHeight] = useState<"0px" | "auto">(isExpanded ? "auto" : "0px");
  const contentRef = useRef<HTMLDivElement>(null);

  useEffect(() => {
    if (!contentRef.current) return;
    if (isExpanded) {
      setHeight(`${contentRef.current.scrollHeight}px`);
    } else {
      setHeight("0px");
    }
  }, [isExpanded, section.description, section.observationGuide]);

  const handleKeyDown = (e: React.KeyboardEvent) => {
    if (e.key === "Enter" || e.key === " ") {
      e.preventDefault();
      onToggle(section.id);
    }
  };

  return (
    <div className="overflow-hidden rounded-lg border border-gray-200 dark:border-gray-700">
      <button
        type="button"
        onClick={() => onToggle(section.id)}
        onKeyDown={handleKeyDown}
        className="flex w-full items-center justify-between bg-white px-3 py-2.5 text-left text-sm font-semibold text-gray-900 transition hover:bg-gray-50 dark:bg-gray-800 dark:text-gray-100 dark:hover:bg-gray-750"
        aria-expanded={isExpanded}
        aria-controls={`explanation-content-${section.id}`}
        id={`explanation-trigger-${section.id}`}
      >
        <span>{section.title}</span>
        <span
          className="text-gray-500 transition-transform dark:text-gray-400"
          aria-hidden
        >
          {isExpanded ? "▼" : "▶"}
        </span>
      </button>
      <div
        id={`explanation-content-${section.id}`}
        role="region"
        aria-labelledby={`explanation-trigger-${section.id}`}
        style={{ height }}
        className="overflow-hidden transition-[height] duration-200 ease-out"
      >
        <div ref={contentRef} className="border-t border-gray-100 px-3 py-2 dark:border-gray-700">
          <ExplanationItem section={section} />
        </div>
      </div>
    </div>
  );
});
