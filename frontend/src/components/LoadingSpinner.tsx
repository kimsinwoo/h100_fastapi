import React from "react";

export const LoadingSpinner: React.FC = () => (
  <div
    className="inline-block h-10 w-10 animate-spin rounded-full border-4 border-solid border-indigo-500 border-r-transparent"
    role="status"
    aria-label="Loading"
  />
);

export const LoadingOverlay: React.FC<{ message?: string }> = ({ message = "Processing..." }) => (
  <div className="fixed inset-0 z-50 flex flex-col items-center justify-center bg-black/60 backdrop-blur-sm">
    <LoadingSpinner />
    <p className="mt-4 text-lg font-medium text-white">{message}</p>
  </div>
);
