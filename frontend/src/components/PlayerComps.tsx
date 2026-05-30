'use client';

import { useEffect, useState, useMemo } from 'react';
import { getComps, CompPlayer } from '../lib/api';
import { formatKtc } from '../lib/format';

interface PlayerCompsProps {
  playerId: string;
  modelId?: string | null;
}

// Pull a few more comparables now that comps are the headline signal.
const COMP_COUNT = 15;

// Linear-interpolated percentile over a pre-sorted ascending array.
function percentile(sorted: number[], p: number): number {
  if (sorted.length === 0) return 0;
  if (sorted.length === 1) return sorted[0];
  const idx = (sorted.length - 1) * p;
  const lo = Math.floor(idx);
  const hi = Math.ceil(idx);
  if (lo === hi) return sorted[lo];
  return sorted[lo] + (sorted[hi] - sorted[lo]) * (idx - lo);
}

export default function PlayerComps({ playerId, modelId }: PlayerCompsProps) {
  const [comps, setComps] = useState<CompPlayer[]>([]);
  const [compAvgPpg, setCompAvgPpg] = useState<number | null>(null);
  const [startKtc, setStartKtc] = useState<number | null>(null);
  const [loading, setLoading] = useState(true);

  useEffect(() => {
    (async () => {
      setLoading(true);
      try {
        const data = await getComps(playerId, COMP_COUNT, modelId);
        setComps(data?.comps ?? []);
        setCompAvgPpg(data?.comp_avg_ppg ?? null);
        setStartKtc(data?.query?.start_ktc ?? null);
      } catch {
        setComps([]);
        setCompAvgPpg(null);
        setStartKtc(null);
      } finally {
        setLoading(false);
      }
    })();
  }, [playerId, modelId]);

  // Comp-implied outlook: project each comp's pct change onto THIS player's
  // start KTC, then take a similarity-weighted center and a p25–p75 range.
  const outlook = useMemo(() => {
    if (comps.length === 0 || startKtc == null || startKtc <= 0) return null;
    const start = startKtc;

    const implied = comps.map((c) => start * (1 + c.pct_change / 100));
    const weights = comps.map((c) => Math.max(c.similarity, 1e-6));
    const wSum = weights.reduce((a, b) => a + b, 0);
    const center = implied.reduce((s, v, i) => s + v * weights[i], 0) / wSum;

    const sorted = [...implied].sort((a, b) => a - b);
    const low = percentile(sorted, 0.25);
    const high = percentile(sorted, 0.75);

    const risers = comps.filter((c) => c.delta_ktc > 0).length;
    const delta = center - start;
    const pct = (delta / start) * 100;

    return { start, center, low, high, lowFull: sorted[0], highFull: sorted[sorted.length - 1], risers, delta, pct };
  }, [comps, startKtc]);

  if (loading) {
    return (
      <div className="glass-card p-6">
        <h3 className="text-base font-semibold text-gray-900 dark:text-white mb-4">
          Historical Comps
        </h3>
        <div className="flex justify-center items-center py-8">
          <div className="w-6 h-6 border-2 border-gray-200 dark:border-gray-600 border-t-blue-600 rounded-full animate-spin" />
        </div>
      </div>
    );
  }

  if (comps.length === 0) return null;

  // Range-bar geometry — domain spans the full comp outcome range plus the start anchor.
  const bar = outlook
    ? (() => {
        const domainMin = Math.min(outlook.lowFull, outlook.start);
        const domainMax = Math.max(outlook.highFull, outlook.start);
        const span = Math.max(domainMax - domainMin, 1);
        const pos = (v: number) => ((v - domainMin) / span) * 100;
        return { pos };
      })()
    : null;

  // Largest |pct_change| in the comp set — used to scale the distribution sparkbars.
  const maxAbsPct = Math.max(1, ...comps.map((c) => Math.abs(c.pct_change)));

  return (
    <div className="glass-card p-6 ring-1 ring-blue-500/10 dark:ring-blue-400/10">
      <div className="flex items-center justify-between mb-1">
        <h3 className="text-base font-semibold text-gray-900 dark:text-white">
          Historical Comps
        </h3>
        <span className="text-[10px] uppercase tracking-wide px-2 py-0.5 rounded-full bg-blue-500/10 dark:bg-blue-500/15 text-blue-600 dark:text-blue-300 font-semibold border border-blue-200/30 dark:border-blue-500/20">
          Primary signal
        </span>
      </div>
      <p className="text-xs text-gray-500 dark:text-gray-400 mb-4">
        The {comps.length} most similar pre-season profiles (KTC, age, prior production) and what
        actually happened to them. Grounded in real outcomes — lean on this over the model estimate.
      </p>

      {/* ── Comp-implied outlook ─────────────────────────────────────── */}
      {outlook && bar && (
        <div className="rounded-xl bg-white/50 dark:bg-white/[0.03] border border-gray-200/40 dark:border-white/[0.06] p-4 mb-4">
          <div className="flex items-end justify-between flex-wrap gap-2 mb-3">
            <div>
              <div className="text-[11px] uppercase tracking-wide text-gray-500 dark:text-gray-400 mb-0.5">
                Comp-implied EOS value
              </div>
              <div className="flex items-baseline gap-2">
                <span className="text-2xl font-extrabold text-gray-900 dark:text-white">
                  {formatKtc(Math.round(outlook.center))}
                </span>
                <span
                  className={`text-sm font-bold ${
                    outlook.delta >= 0
                      ? 'text-green-600 dark:text-green-400'
                      : 'text-red-600 dark:text-red-400'
                  }`}
                >
                  {outlook.delta >= 0 ? '+' : ''}
                  {Math.round(outlook.delta).toLocaleString()} ({outlook.pct >= 0 ? '+' : ''}
                  {outlook.pct.toFixed(0)}%)
                </span>
              </div>
            </div>
            <div className="text-right">
              <div className="text-[11px] uppercase tracking-wide text-gray-500 dark:text-gray-400 mb-0.5">
                Likely range
              </div>
              <div className="text-sm font-semibold text-gray-700 dark:text-gray-300">
                {formatKtc(Math.round(outlook.low))} – {formatKtc(Math.round(outlook.high))}
              </div>
            </div>
          </div>

          {/* Range bar: p25–p75 band, start anchor, weighted center */}
          <div className="relative h-2 rounded-full bg-gray-200/70 dark:bg-white/10 mt-5 mb-2">
            <div
              className="absolute h-2 rounded-full bg-blue-400/50 dark:bg-blue-500/40"
              style={{ left: `${bar.pos(outlook.low)}%`, width: `${Math.max(bar.pos(outlook.high) - bar.pos(outlook.low), 1)}%` }}
            />
            {/* start anchor */}
            <div
              className="absolute -top-1.5 w-px h-5 bg-gray-500 dark:bg-gray-400"
              style={{ left: `${bar.pos(outlook.start)}%` }}
              title={`Current: ${formatKtc(Math.round(outlook.start))}`}
            />
            {/* weighted center */}
            <div
              className="absolute -top-1 w-3 h-3 rounded-full bg-blue-600 dark:bg-blue-400 border-2 border-white dark:border-gray-900 shadow"
              style={{ left: `${bar.pos(outlook.center)}%`, transform: 'translateX(-50%)' }}
              title={`Implied: ${formatKtc(Math.round(outlook.center))}`}
            />
          </div>
          <div className="flex justify-between text-[10px] text-gray-400 dark:text-gray-500">
            <span>now {formatKtc(Math.round(outlook.start))}</span>
            <span>{outlook.risers}/{comps.length} comps rose</span>
          </div>

          {/* Outcome distribution sparkbars */}
          <div className="flex items-end gap-0.5 h-10 mt-3" aria-hidden>
            {[...comps]
              .sort((a, b) => a.pct_change - b.pct_change)
              .map((c) => {
                const h = Math.max(8, (Math.abs(c.pct_change) / maxAbsPct) * 100);
                const up = c.delta_ktc >= 0;
                return (
                  <div
                    key={`${c.player_id}-${c.year}`}
                    className={`flex-1 rounded-sm ${up ? 'bg-green-500/60 dark:bg-green-400/50' : 'bg-red-500/60 dark:bg-red-400/50'}`}
                    style={{ height: `${h}%` }}
                    title={`${c.name} ${c.year}: ${c.pct_change >= 0 ? '+' : ''}${c.pct_change}%`}
                  />
                );
              })}
          </div>
          <div className="flex justify-between text-[10px] text-gray-400 dark:text-gray-500 mt-1">
            <span>biggest fall</span>
            <span>biggest rise</span>
          </div>
        </div>
      )}

      {/* Secondary summary pills */}
      <div className="flex flex-wrap items-center gap-2 mb-4">
        {compAvgPpg != null && (
          <span className="text-xs px-2 py-1 rounded-full bg-blue-500/10 dark:bg-blue-500/15 text-blue-600 dark:text-blue-300 font-medium border border-blue-200/30 dark:border-blue-500/20">
            Projected PPG: {compAvgPpg}
          </span>
        )}
        {outlook && (
          <span className="text-xs px-2 py-1 rounded-full bg-white/40 dark:bg-white/5 text-gray-600 dark:text-gray-400 border border-gray-200/30 dark:border-white/10 font-medium">
            Avg start: {formatKtc(Math.round(comps.reduce((s, c) => s + c.start_ktc, 0) / comps.length))}
          </span>
        )}
      </div>

      <div className="overflow-x-auto">
        <table className="w-full text-sm">
          <thead>
            <tr className="text-left text-xs text-gray-500 dark:text-gray-400 border-b border-gray-200/30 dark:border-white/[0.06]">
              <th className="py-2 pr-3">Player</th>
              <th className="py-2 pr-3 text-right">Year</th>
              <th className="py-2 pr-3 text-right">Age</th>
              <th className="py-2 pr-3 text-right">GP</th>
              <th className="py-2 pr-3 text-right">PPG</th>
              <th className="py-2 pr-3 text-right">Start</th>
              <th className="py-2 pr-3 text-right">End</th>
              <th className="py-2 pr-3 text-right">Change</th>
              <th className="py-2 text-right">Match</th>
            </tr>
          </thead>
          <tbody>
            {comps.map((c, i) => (
              <tr
                key={`${c.player_id}-${c.year}`}
                className={`border-b border-gray-100/30 dark:border-white/[0.03] ${
                  i === 0 ? 'bg-blue-500/[0.04] dark:bg-blue-500/[0.06]' : ''
                }`}
              >
                <td className="py-2.5 pr-3 font-medium text-gray-900 dark:text-white">
                  {c.name}
                  {i === 0 && (
                    <span className="ml-1.5 text-[10px] text-blue-500 dark:text-blue-400 font-semibold">
                      closest
                    </span>
                  )}
                </td>
                <td className="py-2.5 pr-3 text-right text-gray-600 dark:text-gray-400">{c.year}</td>
                <td className="py-2.5 pr-3 text-right text-gray-600 dark:text-gray-400">{c.age}</td>
                <td className="py-2.5 pr-3 text-right text-gray-600 dark:text-gray-400">{c.games_played}</td>
                <td className="py-2.5 pr-3 text-right text-gray-600 dark:text-gray-400">{c.ppg}</td>
                <td className="py-2.5 pr-3 text-right text-gray-600 dark:text-gray-400">{formatKtc(c.start_ktc)}</td>
                <td className="py-2.5 pr-3 text-right font-medium text-gray-900 dark:text-white">{formatKtc(c.end_ktc)}</td>
                <td className={`py-2.5 pr-3 text-right font-bold ${
                  c.delta_ktc >= 0
                    ? 'text-green-600 dark:text-green-400'
                    : 'text-red-600 dark:text-red-400'
                }`}>
                  {c.delta_ktc >= 0 ? '+' : ''}{Math.round(c.delta_ktc).toLocaleString()}
                </td>
                <td className="py-2.5 text-right">
                  <div className="flex items-center justify-end gap-1.5">
                    <div className="w-10 h-1.5 rounded-full bg-gray-200/70 dark:bg-white/10 overflow-hidden">
                      <div
                        className="h-full rounded-full bg-blue-500/70 dark:bg-blue-400/70"
                        style={{ width: `${Math.round(c.similarity * 100)}%` }}
                      />
                    </div>
                    <span className="text-gray-400 dark:text-gray-500 text-xs w-8 text-right">
                      {Math.round(c.similarity * 100)}%
                    </span>
                  </div>
                </td>
              </tr>
            ))}
          </tbody>
        </table>
      </div>
    </div>
  );
}
