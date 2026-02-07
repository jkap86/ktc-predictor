export const KTC_MIN = 1;
export const KTC_MAX = 9999;

/**
 * Clamp a KTC value to valid domain [1, 9999].
 */
export function clampKtc(value: number): number {
  if (!Number.isFinite(value)) return KTC_MIN;
  return Math.max(KTC_MIN, Math.min(KTC_MAX, value));
}

/**
 * Format KTC value for display, clamping to valid domain [1, 9999].
 * Returns "—" for null/undefined values.
 */
export function formatKtc(value?: number | null): string {
  if (value == null) return "—";
  return Math.round(clampKtc(value)).toLocaleString();
}

/**
 * Format KTC value for chart axis ticks, clamping to valid domain.
 */
export function formatKtcTick(value: number): string {
  const clamped = clampKtc(value);
  if (clamped >= 1000) {
    const k = clamped / 1000;
    return `${Number.isInteger(k) ? k : k.toFixed(1)}K`;
  }
  return Math.round(clamped).toString();
}
