'use client';

import { memo, useState, useRef, useCallback } from 'react';

interface EditablePpgProps {
  defaultValue: number | null | undefined;
  overrideValue: number | undefined;
  isLoading: boolean;
  onCommit: (ppg: number) => void;
  onReset: () => void;
}

export const EditablePpg = memo(function EditablePpg({
  defaultValue,
  overrideValue,
  isLoading,
  onCommit,
  onReset,
}: EditablePpgProps) {
  const [editing, setEditing] = useState(false);
  const [inputValue, setInputValue] = useState('');
  const inputRef = useRef<HTMLInputElement>(null);

  const displayValue = overrideValue ?? defaultValue;
  const isOverridden = overrideValue !== undefined;

  const startEdit = useCallback((e: React.MouseEvent) => {
    e.stopPropagation();
    if (displayValue == null) return;
    setInputValue(String(displayValue));
    setEditing(true);
    // Auto-focus + select on next tick
    setTimeout(() => {
      inputRef.current?.focus();
      inputRef.current?.select();
    }, 0);
  }, [displayValue]);

  const commitEdit = useCallback(() => {
    setEditing(false);
    const parsed = parseFloat(inputValue);
    if (isNaN(parsed)) return;
    const clamped = Math.max(0, Math.min(40, Math.round(parsed * 2) / 2));
    // Only commit if different from default
    if (clamped === defaultValue && !isOverridden) return;
    onCommit(clamped);
  }, [inputValue, defaultValue, isOverridden, onCommit]);

  const handleKeyDown = useCallback((e: React.KeyboardEvent) => {
    e.stopPropagation();
    if (e.key === 'Enter') {
      commitEdit();
    } else if (e.key === 'Escape') {
      setEditing(false);
    }
  }, [commitEdit]);

  const handleReset = useCallback((e: React.MouseEvent) => {
    e.stopPropagation();
    onReset();
  }, [onReset]);

  if (editing) {
    return (
      <input
        ref={inputRef}
        type="number"
        step={0.5}
        min={0}
        max={40}
        value={inputValue}
        onChange={(e) => setInputValue(e.target.value)}
        onBlur={commitEdit}
        onKeyDown={handleKeyDown}
        onClick={(e) => e.stopPropagation()}
        className="w-12 text-base sm:text-xs text-right bg-white dark:bg-gray-700 border border-blue-400 rounded px-1 py-0.5 text-gray-900 dark:text-white focus:outline-none focus:ring-1 focus:ring-blue-500 [appearance:textfield] [&::-webkit-outer-spin-button]:appearance-none [&::-webkit-inner-spin-button]:appearance-none"
      />
    );
  }

  if (displayValue == null) {
    return <span className="text-xs text-gray-400 w-12 text-right shrink-0">&mdash;</span>;
  }

  return (
    <span
      onClick={startEdit}
      className={`group relative text-xs w-12 text-right shrink-0 cursor-text border-b border-dashed ${
        isOverridden
          ? 'text-blue-500 dark:text-blue-400 border-blue-300 dark:border-blue-600'
          : 'text-gray-500 dark:text-gray-400 border-gray-300 dark:border-gray-600'
      }`}
    >
      {displayValue}
      {isOverridden && (
        <button
          onClick={handleReset}
          className="absolute -left-3.5 top-1/2 -translate-y-1/2 opacity-0 group-hover:opacity-100 text-gray-400 hover:text-red-400 transition-opacity"
          title="Reset PPG"
        >
          <svg className="w-3 h-3" fill="none" viewBox="0 0 24 24" stroke="currentColor">
            <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M6 18L18 6M6 6l12 12" />
          </svg>
        </button>
      )}
    </span>
  );
});
