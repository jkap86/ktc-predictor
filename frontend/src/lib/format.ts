export const KTC_MIN = 0;
export const KTC_MAX = 9999;

/** Fixed Y-axis domain for all KTC charts [0, 9999] */
export const KTC_Y_DOMAIN: [number, number] = [KTC_MIN, KTC_MAX];

/** Fixed Y-axis ticks for all KTC charts */
export const KTC_Y_TICKS: number[] = [0, 2000, 4000, 6000, 8000, 10000];

/**
 * Clamp a KTC value to valid domain [0, 9999].
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

/**
 * Generate clean Y-axis tick values for KTC charts based on the data range.
 * Uses round increments (250, 500, 1000, 2000) for readability.
 */
export function generateKtcTicks(min: number, max: number): number[] {
  const range = max - min;
  let step: number;
  if (range <= 2000) step = 250;
  else if (range <= 5000) step = 500;
  else if (range <= 10000) step = 1000;
  else step = 2000;

  const start = Math.floor(min / step) * step;
  const ticks: number[] = [];
  for (let t = start; t <= max + step; t += step) {
    if (t >= 0) ticks.push(t);
  }
  return ticks;
}
