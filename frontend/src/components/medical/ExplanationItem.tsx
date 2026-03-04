import React from "react";
import type { MedicalExplanationSection } from "shared/medical/types";

interface ExplanationItemProps {
  section: MedicalExplanationSection;
}

export const ExplanationItem = React.memo(function ExplanationItem({
  section,
}: ExplanationItemProps) {
  return (
    <div className="space-y-2 pl-2">
      <p className="text-sm text-gray-700 dark:text-gray-300">
        {section.description}
      </p>
      <p className="text-xs text-gray-600 dark:text-gray-400">
        <span className="font-medium">관찰: </span>
        {section.observationGuide}
      </p>
    </div>
  );
});
