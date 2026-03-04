import { create } from "zustand";
import type { MedicalResultUIState } from "shared/medical/types";

interface MedicalResultStore extends MedicalResultUIState {
  toggleExplanation: (id: string) => void;
  setCopiedQuestionId: (id: string | null) => void;
  setExpandedExplanationIds: (ids: Set<string>) => void;
}

export const useMedicalResultStore = create<MedicalResultStore>((set) => ({
  expandedExplanationIds: new Set<string>(),
  copiedQuestionId: null,
  toggleExplanation: (id) =>
    set((s) => {
      const next = new Set(s.expandedExplanationIds);
      if (next.has(id)) next.delete(id);
      else next.add(id);
      return { expandedExplanationIds: next };
    }),
  setCopiedQuestionId: (id) => set({ copiedQuestionId: id }),
  setExpandedExplanationIds: (ids) => set({ expandedExplanationIds: ids }),
}));
