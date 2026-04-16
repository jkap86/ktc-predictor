export const KTC_MIN = 0;
export const KTC_MAX = 9999;

export const KTC_Y_DOMAIN: [number, number] = [KTC_MIN, KTC_MAX];
export const KTC_Y_TICKS: number[] = [0, 2000, 4000, 6000, 8000, 10000];

export function clampKtc(value: number): number {
  if (!Number.isFinite(value)) return KTC_MIN;
  return Math.max(KTC_MIN, Math.min(KTC_MAX, value));
}

export function formatKtc(value?: number | null): string {
  if (value == null) return "\u2014";
  return Math.round(clampKtc(value)).toLocaleString();
}

export function formatKtcTick(value: number): string {
  const clamped = clampKtc(value);
  if (clamped >= 1000) {
    const k = clamped / 1000;
    return `${Number.isInteger(k) ? k : k.toFixed(1)}K`;
  }
  return Math.round(clamped).toString();
}
