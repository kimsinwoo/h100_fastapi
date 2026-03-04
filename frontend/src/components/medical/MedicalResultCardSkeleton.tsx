import React from "react";

export const MedicalResultCardSkeleton = React.memo(
  function MedicalResultCardSkeleton() {
    return (
      <div
        className="animate-pulse overflow-hidden rounded-xl border border-gray-200 bg-white dark:border-gray-700 dark:bg-gray-800"
        aria-busy="true"
        aria-label="로딩 중"
      >
        <div className="p-4 space-y-4">
          <div className="h-6 w-16 rounded-full bg-gray-200 dark:bg-gray-600" />
          <div className="space-y-2">
            <div className="h-4 w-full rounded bg-gray-200 dark:bg-gray-600" />
            <div className="h-4 w-4/5 rounded bg-gray-200 dark:bg-gray-600" />
          </div>
          <div className="space-y-2">
            {[1, 2, 3].map((i) => (
              <div
                key={i}
                className="h-12 rounded-lg bg-gray-100 dark:bg-gray-700"
              />
            ))}
          </div>
          <div className="h-24 rounded-lg bg-red-50 dark:bg-red-950/30" />
          <div className="flex gap-2">
            <div className="h-10 w-24 rounded-lg bg-gray-200 dark:bg-gray-600" />
            <div className="h-10 w-24 rounded-lg bg-gray-200 dark:bg-gray-600" />
          </div>
        </div>
      </div>
    );
  }
);
