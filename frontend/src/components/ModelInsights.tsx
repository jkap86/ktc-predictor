'use client';

import { useEffect, useState } from 'react';
import { getModelDiagnostics, ModelDiagnostics } from '../lib/api';
import { formatKtc } from '../lib/format';
import { useModel } from '../context/ModelContext';

const POS_ORDER = ['QB', 'RB', 'WR', 'TE'];

const TIER_LABELS: Record<string, string> = {
  '0-2k': '<2K',
  '2k-4k': '2-4K',
  '4k-6k': '4-6K',
  '6k+': '6K+',
};

function BiasBar({ bias, maxBias }: { bias: number; maxBias: number }) {
  const pct = Math.min(Math.abs(bias) / maxBias * 50, 50);
  const isPositive = bias > 0;
  return (
    <div className="flex items-center gap-1.5 w-full">
      <div className="flex-1 h-2 rounded-full bg-white/10 relative overflow-hidden">
        <div
          className={`absolute h-full rounded-full ${isPositive ? 'bg-amber-400/60' : 'bg-blue-400/60'}`}
          style={{
            width: `${pct}%`,
            left: isPositive ? '50%' : `${50 - pct}%`,
          }}
        />
        <div className="absolute left-1/2 top-0 bottom-0 w-px bg-white/20" />
      </div>
      <span className={`text-[10px] font-mono w-12 text-right ${
        Math.abs(bias) < 100 ? 'text-green-400' : Math.abs(bias) < 300 ? 'text-amber-400' : 'text-red-400'
      }`}>
        {bias > 0 ? '+' : ''}{Math.round(bias)}
      </span>
    </div>
  );
}

export default function ModelInsights() {
  const { selectedModelId } = useModel();
  const [data, setData] = useState<ModelDiagnostics | null>(null);
  const [loading, setLoading] = useState(true);
  const [expanded, setExpanded] = useState(false);

  useEffect(() => {
    (async () => {
      setLoading(true);
      const d = await getModelDiagnostics(selectedModelId);
      setData(d);
      setLoading(false);
    })();
  }, [selectedModelId]);

  if (loading) {
    return (
      <div className="glass-card p-4">
        <h3 className="text-sm font-semibold text-gray-900 dark:text-white mb-3">Model Insights</h3>
        <div className="flex justify-center py-4">
          <div className="w-5 h-5 border-2 border-gray-200 dark:border-gray-600 border-t-blue-600 rounded-full animate-spin" />
        </div>
      </div>
    );
  }

  if (!data?.diagnostics?.positions) return null;

  const positions = data.diagnostics.positions;
  const metrics = data.metrics;

  // Find max bias for scaling bars
  const allBiases: number[] = [];
  for (const pos of POS_ORDER) {
    const p = positions[pos];
    if (!p) continue;
    allBiases.push(Math.abs(p.bias));
    for (const t of Object.values(p.tiers)) {
      allBiases.push(Math.abs(t.bias));
    }
  }
  const maxBias = Math.max(...allBiases, 100);

  // Generate strength/weakness insights
  const insights: { type: 'strength' | 'weakness'; text: string }[] = [];

  for (const pos of POS_ORDER) {
    const p = positions[pos];
    if (!p) continue;
    const m = metrics[pos];

    if (m && m.r2 >= 0.92) {
      insights.push({ type: 'strength', text: `${pos}: R\u00B2 = ${(m.r2 * 100).toFixed(1)}% variance explained` });
    }

    for (const [tkey, tier] of Object.entries(p.tiers)) {
      const label = TIER_LABELS[tkey] || tkey;
      if (Math.abs(tier.bias) < 50 && tier.mae < 350) {
        insights.push({ type: 'strength', text: `${pos} ${label}: MAE ${formatKtc(tier.mae)}, near-zero bias` });
      } else if (Math.abs(tier.bias) > 400) {
        const dir = tier.bias > 0 ? 'over' : 'under';
        insights.push({ type: 'weakness', text: `${pos} ${label}: ${dir}predicts by ~${formatKtc(Math.abs(tier.bias))}` });
      } else if (tier.mae > 800) {
        insights.push({ type: 'weakness', text: `${pos} ${label}: high error (MAE ${formatKtc(tier.mae)})` });
      }
    }
  }

  // Sort: strengths first
  insights.sort((a, b) => (a.type === b.type ? 0 : a.type === 'strength' ? -1 : 1));

  return (
    <div className="glass-card p-4">
      <div className="flex items-center justify-between mb-3">
        <h3 className="text-sm font-semibold text-gray-900 dark:text-white">Model Insights</h3>
        <button
          onClick={() => setExpanded((e) => !e)}
          className="text-xs text-blue-500 hover:text-blue-400 transition-colors"
        >
          {expanded ? 'Less' : 'More'}
        </button>
      </div>

      {/* Summary cards */}
      <div className="grid grid-cols-4 gap-2 mb-4">
        {POS_ORDER.map((pos) => {
          const m = metrics[pos];
          const p = positions[pos];
          if (!m || !p) return null;
          return (
            <div key={pos} className="rounded-xl bg-white/5 border border-white/[0.06] p-2.5 text-center" style={{boxShadow: 'inset 0 1px 0 rgba(255,255,255,0.03)'}}>
              <div className="text-xs font-bold text-gray-400 dark:text-gray-500 mb-1">{pos}</div>
              <div className="text-lg font-bold text-white">{formatKtc(m.mae)}</div>
              <div className="text-[10px] text-gray-500">MAE</div>
              <div className="text-xs text-gray-400 mt-0.5">{(m.r2 * 100).toFixed(0)}% R&sup2;</div>
            </div>
          );
        })}
      </div>

      {/* Insights list */}
      <div className="space-y-1.5 mb-3">
        {insights.slice(0, expanded ? undefined : 4).map((ins, i) => (
          <div key={i} className="flex items-start gap-2 text-xs">
            <span className={`mt-0.5 shrink-0 ${ins.type === 'strength' ? 'text-green-400' : 'text-amber-400'}`}>
              {ins.type === 'strength' ? '+' : '!'}
            </span>
            <span className="text-gray-300">{ins.text}</span>
          </div>
        ))}
      </div>

      {/* Tier breakdown table */}
      {expanded && (
        <div className="mt-4">
          <div className="text-xs font-semibold text-gray-400 dark:text-gray-500 mb-2 uppercase tracking-wider">Bias by KTC Tier</div>
          <div className="space-y-3">
            {POS_ORDER.map((pos) => {
              const p = positions[pos];
              if (!p) return null;
              return (
                <div key={pos}>
                  <div className="text-xs font-medium text-gray-300 mb-1">{pos} <span className="text-gray-500">(MAE {formatKtc(p.mae)})</span></div>
                  <div className="space-y-1">
                    {Object.entries(p.tiers).map(([tkey, tier]) => (
                      <div key={tkey} className="flex items-center gap-2 text-[11px]">
                        <span className="w-10 text-gray-500 shrink-0">{TIER_LABELS[tkey] || tkey}</span>
                        <BiasBar bias={tier.bias} maxBias={maxBias} />
                        <span className="text-gray-500 w-10 text-right shrink-0">{tier.n}</span>
                      </div>
                    ))}
                  </div>
                </div>
              );
            })}
          </div>
          <div className="flex items-center gap-3 mt-3 text-[10px] text-gray-500">
            <span className="flex items-center gap-1"><span className="w-3 h-1.5 rounded-full bg-blue-400/60 inline-block"></span> Underpredicts</span>
            <span className="flex items-center gap-1"><span className="w-3 h-1.5 rounded-full bg-amber-400/60 inline-block"></span> Overpredicts</span>
            <span className="flex items-center gap-1"><span className="text-green-400">0</span> = unbiased</span>
          </div>
        </div>
      )}
    </div>
  );
}
