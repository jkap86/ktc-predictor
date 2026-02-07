/**
 * Format KTC value for display, clamping to valid domain [1, 9999].
 * Returns "—" for null/undefined values.
 */
export function formatKtc(value?: number | null): string {
  if (value == null) return "—";
  const clamped = Math.max(1, Math.min(9999, Math.round(value)));
  return clamped.toLocaleString();
}
