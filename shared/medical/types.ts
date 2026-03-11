/**
 * Medical AI Chat – domain model (shared web + native).
 * No parsing here; assume structured data is already parsed.
 */

export type MedicalSeverity = "low" | "medium" | "high" | "critical";

export interface MedicalExplanationSection {
  id: string;
  title: string;
  description: string;
  observationGuide: string;
}

export interface EmergencyCriteria {
  items: string[];
}

export interface FollowUpQuestion {
  id: string;
  question: string;
}

export interface RecommendedActions {
  primary: string;
  secondary?: string;
}

export interface MedicalChatStructured {
  summary: string;
  severity: MedicalSeverity;
  explanations: MedicalExplanationSection[];
  emergencyCriteria: EmergencyCriteria;
  followUpQuestions: FollowUpQuestion[];
  recommendedActions: RecommendedActions;
}

/** For accordion / UI state only; not part of API payload. */
export interface MedicalResultUIState {
  expandedExplanationIds: Set<string>;
  copiedQuestionId: string | null;
}
