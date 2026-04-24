'use client';

import { useEffect, useState, useRef, useMemo } from 'react';
import {
  ResponsiveContainer,
  Area,
  Line,
  XAxis,
  YAxis,
  CartesianGrid,
  Tooltip,
  ReferenceLine,
  ReferenceDot,
  ComposedChart,
} from 'recharts';
import { predictEosBatch, predictPlayerWhatIfBatch } from '../lib/api';
import { formatKtc } from '../lib/format';

interface PlayerCurveConfig {
  position: string;
  startKtc: number;
  age?: number;
  draftPick?: number;
  yearsRemaining?: number;
  weeksMissed?: number;
}

interface WhatIfChartProps {
  position: string;
  startKtc: number;
  gamesPlayed: number;
  currentPpg: number;
  modelId: string | null;
  age?: number;
  draftPick?: number;
  yearsRemaining?: number;
  weeksMissed?: number;
  /** Player ID for player-aware predictions (uses full feature context) */
  playerId?: string;
  /** Optional comparison player config */
  compare?: PlayerCurveConfig & { name: string; playerId?: string };
}

interface ChartPoint {
  ppg: number;
  predictedEos: number | null;
  low: number | null;
  high: number | null;
  compareEos: number | null;
  compareLow: number | null;
  compareHigh: number | null;
  // For stacked area band rendering: [low, high] tuple
  bandRange: [number, number] | null;
  compareBandRange: [number, number] | null;
}

const PPG_STEPS = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25];

async function fetchCurveForPlayer(
  config: PlayerCurveConfig,
  gamesPlayed: number,
  modelId: string | null,
  playerId?: string,
): Promise<Map<number, { eos: number | null; low: number | null; high: number | null }>> {
  const results = new Map<number, { eos: number | null; low: number | null; high: number | null }>();

  try {
    // Use player-aware endpoint when playerId is available (includes full feature context)
    const batch = playerId
      ? await predictPlayerWhatIfBatch(
          playerId,
          { games_played: gamesPlayed, ppg_values: PPG_STEPS },
          modelId,
        )
      : await predictEosBatch(
          {
            position: config.position,
            start_ktc: config.startKtc,
            games_played: gamesPlayed,
            ppg_values: PPG_STEPS,
            age: config.age,
            draft_pick: config.draftPick,
            years_remaining: config.yearsRemaining,
            weeks_missed: config.weeksMissed,
          },
          modelId,
        );

    if (batch) {
      batch.predictions.forEach((pred, i) => {
        const ppg = PPG_STEPS[i];
        results.set(ppg, {
          eos: pred.predicted_end_ktc || null,
          low: pred.low_end_ktc ?? null,
          high: pred.high_end_ktc ?? null,
        });
      });
    }
  } catch {
    for (const ppg of PPG_STEPS) {
      results.set(ppg, { eos: null, low: null, high: null });
    }
  }

  return results;
}

export default function WhatIfChart({
  position,
  startKtc,
  gamesPlayed,
  currentPpg,
  modelId,
  age,
  draftPick,
  yearsRemaining,
  weeksMissed,
  playerId,
  compare,
}: WhatIfChartProps) {
  const [data, setData] = useState<ChartPoint[]>([]);
  const [loading, setLoading] = useState(false);
  const abortRef = useRef(0);

  const primaryConfig = useMemo<PlayerCurveConfig>(() => ({
    position, startKtc, age, draftPick, yearsRemaining, weeksMissed,
  }), [position, startKtc, age, draftPick, yearsRemaining, weeksMissed]);

  useEffect(() => {
    const generation = ++abortRef.current;
    setData([]);

    async function fetchCurves() {
      setLoading(true);

      const fetches: [Promise<Map<number, { eos: number | null; low: number | null; high: number | null }>>, Promise<Map<number, { eos: number | null; low: number | null; high: number | null }>> | null] = [
        fetchCurveForPlayer(primaryConfig, gamesPlayed, modelId, playerId),
        compare ? fetchCurveForPlayer(compare, gamesPlayed, modelId, compare.playerId) : null,
      ];

      const [primaryMap, compareMap] = await Promise.all([
        fetches[0],
        fetches[1] ?? Promise.resolve(null),
      ]);

      if (generation !== abortRef.current) return;

      const merged: ChartPoint[] = PPG_STEPS.map((ppg) => {
        const p = primaryMap.get(ppg);
        const c = compareMap?.get(ppg);
        return {
          ppg,
          predictedEos: p?.eos ?? null,
          low: p?.low ?? null,
          high: p?.high ?? null,
          compareEos: c?.eos ?? null,
          compareLow: c?.low ?? null,
          compareHigh: c?.high ?? null,
          bandRange: p?.low != null && p?.high != null ? [p.low, p.high] : null,
          compareBandRange: c?.low != null && c?.high != null ? [c.low, c.high] : null,
        };
      });

      setData(merged);
      setLoading(false);
    }

    fetchCurves();
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [
    position, startKtc, gamesPlayed, modelId, age, draftPick, yearsRemaining, weeksMissed, playerId,
    compare?.position, compare?.startKtc, compare?.age, compare?.playerId,
  ]);

  const currentPoint = useMemo(() => data.reduce<ChartPoint | null>((best, pt) => {
    if (pt.predictedEos === null) return best;
    if (!best) return pt;
    return Math.abs(pt.ppg - currentPpg) < Math.abs(best.ppg - currentPpg) ? pt : best;
  }, null), [data, currentPpg]);

  const hasBands = useMemo(() => data.some((d) => d.low !== null && d.high !== null), [data]);
  const hasCompare = useMemo(() => data.some((d) => d.compareEos !== null), [data]);

  if (loading && data.length === 0) {
    return (
      <div className="flex justify-center items-center h-64">
        <div className="w-6 h-6 border-2 border-gray-200 dark:border-gray-600 border-t-blue-600 rounded-full animate-spin" />
      </div>
    );
  }

  if (data.length === 0) return null;

  return (
    <div className="mt-6">
      <div className="flex items-center justify-between mb-2">
        <h4 className="text-sm font-medium text-gray-700 dark:text-gray-300">
          Predicted EOS by PPG
        </h4>
        <div className="flex items-center gap-3">
          {hasCompare && compare && (
            <div className="flex items-center gap-1.5 text-xs text-gray-500 dark:text-gray-400">
              <span className="inline-block w-3 h-0.5 bg-blue-500 rounded"></span>
              Primary
              <span className="inline-block w-3 h-0.5 bg-orange-500 rounded ml-2"></span>
              {compare.name}
            </div>
          )}
          {loading && (
            <div className="w-4 h-4 border-2 border-gray-200 dark:border-gray-600 border-t-blue-600 rounded-full animate-spin" />
          )}
        </div>
      </div>
      <ResponsiveContainer width="100%" height={280}>
        <ComposedChart data={data} margin={{ top: 10, right: 10, bottom: 10, left: 10 }}>
          <CartesianGrid strokeDasharray="3 3" stroke="rgba(128,128,128,0.15)" />
          <XAxis
            dataKey="ppg"
            type="number"
            domain={[0, 25]}
            tick={{ fontSize: 12, fill: '#9ca3af' }}
            label={{ value: 'PPG', position: 'insideBottom', offset: -5, fontSize: 12, fill: '#9ca3af' }}
          />
          <YAxis
            tick={{ fontSize: 12, fill: '#9ca3af' }}
            tickFormatter={(v: number) => formatKtc(v)}
            width={55}
          />
          <Tooltip
            contentStyle={{
              backgroundColor: 'rgba(31,41,55,0.95)',
              border: 'none',
              borderRadius: '8px',
              fontSize: 13,
            }}
            labelStyle={{ color: '#d1d5db', fontWeight: 600 }}
            labelFormatter={(v) => `PPG: ${v}`}
            content={({ active, payload, label }) => {
              if (!active || !payload?.length) return null;
              const pt = data.find((d) => d.ppg === label);
              if (!pt) return null;
              return (
                <div style={{ backgroundColor: 'rgba(31,41,55,0.95)', border: 'none', borderRadius: 8, padding: '8px 12px', fontSize: 13 }}>
                  <p style={{ color: '#d1d5db', fontWeight: 600, marginBottom: 4 }}>PPG: {label}</p>
                  {pt.predictedEos != null && (
                    <p style={{ color: '#3b82f6' }}>Predicted EOS: {formatKtc(pt.predictedEos)}</p>
                  )}
                  {pt.low != null && pt.high != null && (
                    <p style={{ color: '#93c5fd' }}>Range: {formatKtc(pt.low)} – {formatKtc(pt.high)}</p>
                  )}
                  {pt.compareEos != null && (
                    <p style={{ color: '#f97316' }}>{compare?.name ?? 'Compare'}: {formatKtc(pt.compareEos)}</p>
                  )}
                  {pt.compareLow != null && pt.compareHigh != null && (
                    <p style={{ color: '#fdba74' }}>Range: {formatKtc(pt.compareLow)} – {formatKtc(pt.compareHigh)}</p>
                  )}
                </div>
              );
            }}
          />

          {/* Start KTC reference line (primary) */}
          <ReferenceLine
            y={startKtc}
            stroke="#9ca3af"
            strokeDasharray="6 3"
            label={{
              value: `Current: ${formatKtc(startKtc)}`,
              position: 'insideTopRight',
              fontSize: 11,
              fill: '#9ca3af',
            }}
          />

          {/* Comparison player start KTC reference */}
          {hasCompare && compare && (
            <ReferenceLine
              y={compare.startKtc}
              stroke="#f97316"
              strokeDasharray="6 3"
              strokeOpacity={0.5}
            />
          )}

          {/* Shaded confidence bands using Area with [low, high] range */}
          {hasBands && (
            <Area
              dataKey="bandRange"
              stroke="none"
              fill="rgba(59,130,246,0.25)"
              isAnimationActive={false}
              name="bandRange"
              tooltipType="none"
            />
          )}
          {hasCompare && (
            <Area
              dataKey="compareBandRange"
              stroke="none"
              fill="rgba(249,115,22,0.22)"
              isAnimationActive={false}
              name="compareBandRange"
              tooltipType="none"
            />
          )}

          {/* Primary prediction curve */}
          <Line
            dataKey="predictedEos"
            stroke="#3b82f6"
            strokeWidth={2.5}
            dot={false}
            isAnimationActive={false}
            name="predictedEos"
          />

          {/* Band values shown via custom tooltip, no hidden lines needed */}

          {/* Comparison prediction curve */}
          {hasCompare && (
            <Line
              dataKey="compareEos"
              stroke="#f97316"
              strokeWidth={2.5}
              dot={false}
              isAnimationActive={false}
              name="compareEos"
            />
          )}

          {/* Comparison band values shown via custom tooltip, no hidden lines needed */}

          {/* Current PPG marker (primary) */}
          {currentPoint && currentPoint.predictedEos !== null && (
            <ReferenceDot
              x={currentPoint.ppg}
              y={currentPoint.predictedEos}
              r={6}
              fill="#3b82f6"
              stroke="white"
              strokeWidth={2}
            />
          )}

          {/* Current PPG marker (comparison) */}
          {currentPoint && currentPoint.compareEos !== null && (
            <ReferenceDot
              x={currentPoint.ppg}
              y={currentPoint.compareEos}
              r={6}
              fill="#f97316"
              stroke="white"
              strokeWidth={2}
            />
          )}
        </ComposedChart>
      </ResponsiveContainer>
    </div>
  );
}
