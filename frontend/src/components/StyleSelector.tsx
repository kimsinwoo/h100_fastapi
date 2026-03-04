import React from "react";
import type { StylesResponse } from "../types/api";

interface StyleSelectorProps {
  /** API /api/styles 응답 (key -> 표시명). 없으면 빈 목록 */
  styles: StylesResponse | null;
  selected: string;
  onSelect: (style: string) => void;
  disabled?: boolean;
}

export const StyleSelector: React.FC<StyleSelectorProps> = ({
  styles,
  selected,
  onSelect,
  disabled = false,
}) => {
  const entries = styles ? Object.entries(styles) : [];
  return (
    <div className="grid grid-cols-2 gap-2 sm:grid-cols-4 md:grid-cols-5">
      {entries.length === 0 && (
        <p className="col-span-full text-sm text-gray-500">스타일 목록 불러오는 중...</p>
      )}
      {entries.map(([key, label]) => (
        <button
          key={key}
          type="button"
          disabled={disabled}
          onClick={() => onSelect(key)}
          className={`
            rounded-lg border-2 px-3 py-2 text-sm font-medium transition
            ${selected === key
              ? "border-indigo-600 bg-indigo-600 text-white"
              : "border-gray-200 bg-white text-gray-700 hover:border-indigo-400"}
            ${disabled ? "cursor-not-allowed opacity-60" : ""}
          `}
        >
          {label || key}
        </button>
      ))}
    </div>
  );
};
