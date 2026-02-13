import React, { useCallback, useState } from "react";

type DropHandler = (file: File) => void;

interface ImageUploaderProps {
  onFileSelect: DropHandler;
  selectedFile: File | null;
  disabled?: boolean;
}

export const ImageUploader: React.FC<ImageUploaderProps> = ({
  onFileSelect,
  selectedFile,
  disabled = false,
}) => {
  const [isDragOver, setIsDragOver] = useState(false);

  const validateAndSet = useCallback(
    (file: File) => {
      if (!file.type.startsWith("image/")) return;
      onFileSelect(file);
    },
    [onFileSelect]
  ) as DropHandler;

  const handleDrop = useCallback(
    (e: React.DragEvent<HTMLDivElement>) => {
      e.preventDefault();
      setIsDragOver(false);
      if (disabled) return;
      const file = e.dataTransfer.files[0];
      if (file) validateAndSet(file);
    },
    [disabled, validateAndSet]
  );

  const handleDragOver = useCallback((e: React.DragEvent<HTMLDivElement>) => {
    e.preventDefault();
    e.stopPropagation();
    setIsDragOver(true);
  }, []);

  const handleDragLeave = useCallback((e: React.DragEvent<HTMLDivElement>) => {
    e.preventDefault();
    e.stopPropagation();
    setIsDragOver(false);
  }, []);

  const handleChange = useCallback(
    (e: React.ChangeEvent<HTMLInputElement>) => {
      const file = e.target.files?.[0];
      if (file) validateAndSet(file);
      e.target.value = "";
    },
    [validateAndSet]
  );

  return (
    <div
      onDrop={handleDrop}
      onDragOver={handleDragOver}
      onDragLeave={handleDragLeave}
      className={`
        relative flex min-h-[200px] cursor-pointer flex-col items-center justify-center rounded-xl border-2 border-dashed p-6 transition-colors
        ${disabled ? "cursor-not-allowed opacity-60" : ""}
        ${isDragOver ? "border-indigo-500 bg-indigo-50" : "border-gray-300 bg-gray-50 hover:border-gray-400"}
      `}
    >
      <input
        type="file"
        accept="image/*"
        onChange={handleChange}
        disabled={disabled}
        className="absolute inset-0 cursor-pointer opacity-0"
        aria-label="Upload image"
      />
      {selectedFile ? (
        <p className="text-sm font-medium text-gray-700">{selectedFile.name}</p>
      ) : (
        <p className="text-center text-sm text-gray-500">
          Drag & drop an image here, or click to select
        </p>
      )}
    </div>
  );
};
