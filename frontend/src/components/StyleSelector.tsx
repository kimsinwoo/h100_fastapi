import React from "react";
import type { StyleKey } from "../types/api";

const STYLE_KEYS: StyleKey[] = [
  "anime",
  "realistic",
  "watercolor",
  "cyberpunk",
  "oil painting",
  "sketch",
  "cinematic",
  "fantasy art",
  "pixel art",
  "3d render",
];

const STYLE_LABELS: Record<StyleKey, string> = {
  anime: "Anime",
  realistic: "Realistic",
  watercolor: "Watercolor",
  cyberpunk: "Cyberpunk",
  "oil painting": "Oil Painting",
  sketch: "Sketch",
  cinematic: "Cinematic",
  "fantasy art": "Fantasy Art",
  "pixel art": "Pixel Art",
  "3d render": "3D Render",
};

interface StyleSelectorProps {
  selected: string;
  onSelect: (style: string) => void;
  disabled?: boolean;
}

export const StyleSelector: React.FC<StyleSelectorProps> = ({
  selected,
  onSelect,
  disabled = false,
}) => (
  <div className="grid grid-cols-2 gap-2 sm:grid-cols-5">
    {STYLE_KEYS.map((key) => (
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
        {STYLE_LABELS[key]}
      </button>
    ))}
  </div>
);
