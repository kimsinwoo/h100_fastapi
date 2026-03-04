import React from "react";
import { MedicalResultCard, MedicalResultCardSkeleton } from "../components/medical";
import { mockMedicalStructured } from "../components/medical/mockMedicalData";

export default function MedicalResultExample() {
  const [loading, setLoading] = React.useState(false);
  const handlePrimary = () => window.alert("Primary action");
  const handleSecondary = () => window.alert("Secondary action");

  return (
    <div className="min-h-screen bg-gray-100 p-4 dark:bg-gray-900 md:p-6">
      <div className="mx-auto max-w-2xl space-y-6">
        <h1 className="text-xl font-bold text-gray-900 dark:text-white">
          의료 AI 응답 예시
        </h1>
        {loading ? (
          <MedicalResultCardSkeleton />
        ) : (
          <MedicalResultCard
            data={mockMedicalStructured}
            onPrimaryAction={handlePrimary}
            onSecondaryAction={handleSecondary}
            actionDisabled={false}
          />
        )}
      </div>
    </div>
  );
}
