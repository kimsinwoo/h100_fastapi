import React from "react";
import { cva, type VariantProps } from "class-variance-authority";
import type { RecommendedActions } from "shared/medical/types";

const actionButton = cva(
  "rounded-lg px-4 py-2.5 text-sm font-semibold transition focus:outline-none focus:ring-2 focus:ring-offset-2 disabled:opacity-50 disabled:pointer-events-none",
  {
    variants: {
      variant: {
        primary:
          "bg-indigo-600 text-white hover:bg-indigo-700 focus:ring-indigo-500 dark:bg-indigo-500 dark:hover:bg-indigo-600",
        danger:
          "bg-red-600 text-white hover:bg-red-700 focus:ring-red-500 dark:bg-red-500 dark:hover:bg-red-600",
      },
    },
    defaultVariants: { variant: "primary" },
  }
);

type ButtonProps = VariantProps<typeof actionButton> & {
  label: string;
  onClick: () => void;
  disabled?: boolean;
  "aria-label"?: string;
};

const ActionButton = React.memo(function ActionButton({
  variant,
  label,
  onClick,
  disabled,
  "aria-label": ariaLabel,
}: ButtonProps) {
  return (
    <button
      type="button"
      className={actionButton({ variant })}
      onClick={onClick}
      disabled={disabled}
      aria-label={ariaLabel ?? label}
    >
      {label}
    </button>
  );
});

interface ActionButtonGroupProps {
  actions: RecommendedActions;
  onPrimary: () => void;
  onSecondary?: () => void;
  disabled?: boolean;
}

export const ActionButtonGroup = React.memo(function ActionButtonGroup({
  actions,
  onPrimary,
  onSecondary,
  disabled = false,
}: ActionButtonGroupProps) {
  const secondary = actions.secondary;

  return (
    <div
      className="flex flex-wrap gap-2"
      role="group"
      aria-label="권장 행동"
    >
      <ActionButton
        variant="primary"
        label={actions.primary}
        onClick={onPrimary}
        disabled={disabled}
      />
      {secondary != null && secondary !== "" && (
        <ActionButton
          variant="danger"
          label={secondary}
          onClick={onSecondary ?? (() => {})}
          disabled={disabled}
        />
      )}
    </div>
  );
});
