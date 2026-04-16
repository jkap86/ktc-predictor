'use client';

import { useEffect, useState, useRef } from 'react';
import {
  ResponsiveContainer,
  Line,
  XAxis,
  YAxis,
  CartesianGrid,
  Tooltip,
  ReferenceLine,
  ReferenceDot,
  ComposedChart,
} from 'recharts';
import { predictEos } from '../lib/api';
import { formatKtc } from '../lib/format';

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
}

interface ChartPoint {
  ppg: number;
  predictedEos: number | null;
  low: number | null;
  high: number | null;
}

const PPG_STEPS = [0, 2, 4, 6, 8, 10, 12, 14, 16, 18, 20, 22, 24, 26, 28, 30, 32, 34, 36, 38, 40];

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
}: WhatIfChartProps) {
  const [data, setData] = useState<ChartPoint[]>([]);
  const [loading, setLoading] = useState(false);
  const abortRef = useRef(0);

  useEffect(() => {
    const generation = ++abortRef.current;

    async function fetchCurve() {
      setLoading(true);
      const results: ChartPoint[] = [];

      // Fire all requests in parallel
      const promises = PPG_STEPS.map(async (ppg) => {
        const result = await predictEos(
          {
            position,
            start_ktc: startKtc,
            games_played: gamesPlayed,
            ppg,
            age,
            draft_pick: draftPick,
            years_remaining: yearsRemaining,
            weeks_missed: weeksMissed,
          },
          modelId,
        );
        return {
          ppg,
          predictedEos: result?.predicted_end_ktc ?? null,
          low: result?.low_end_ktc ?? null,
          high: result?.high_end_ktc ?? null,
        };
      });

      const settled = await Promise.all(promises);
      if (generation !== abortRef.current) return; // stale

      settled.sort((a, b) => a.ppg - b.ppg);
      setData(settled);
      setLoading(false);
    }

    fetchCurve();
  }, [position, startKtc, gamesPlayed, modelId, age, draftPick, yearsRemaining, weeksMissed]);

  // Find the data point closest to currentPpg for the reference dot
  const currentPoint = data.reduce<ChartPoint | null>((best, pt) => {
    if (pt.predictedEos === null) return best;
    if (!best) return pt;
    return Math.abs(pt.ppg - currentPpg) < Math.abs(best.ppg - currentPpg) ? pt : best;
  }, null);

  const hasBands = data.some((d) => d.low !== null && d.high !== null);

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
          <span className="text-xs text-gray-400 dark:text-gray-500 ml-2">
            ({gamesPlayed} games)
          </span>
        </h4>
        {loading && (
          <div className="w-4 h-4 border-2 border-gray-200 dark:border-gray-600 border-t-blue-600 rounded-full animate-spin" />
        )}
      </div>
      <ResponsiveContainer width="100%" height={280}>
        <ComposedChart data={data} margin={{ top: 10, right: 10, bottom: 10, left: 10 }}>
          <CartesianGrid strokeDasharray="3 3" stroke="rgba(128,128,128,0.15)" />
          <XAxis
            dataKey="ppg"
            type="number"
            domain={[0, 40]}
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
            formatter={(value, name) => {
              const label =
                name === 'predictedEos'
                  ? 'Predicted EOS'
                  : name === 'high'
                    ? 'p80'
                    : name === 'low'
                      ? 'p20'
                      : String(name);
              return [formatKtc(Number(value)), label];
            }}
          />

          {/* Start KTC reference line */}
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

          {/* Main prediction curve */}
          <Line
            dataKey="predictedEos"
            stroke="#3b82f6"
            strokeWidth={2.5}
            dot={false}
            isAnimationActive={false}
          />

          {/* Band boundaries */}
          {hasBands && (
            <>
              <Line dataKey="high" stroke="#93c5fd" strokeWidth={1} strokeDasharray="4 2" dot={false} isAnimationActive={false} />
              <Line dataKey="low" stroke="#93c5fd" strokeWidth={1} strokeDasharray="4 2" dot={false} isAnimationActive={false} />
            </>
          )}

          {/* Current PPG marker */}
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
        </ComposedChart>
      </ResponsiveContainer>
    </div>
  );
}
