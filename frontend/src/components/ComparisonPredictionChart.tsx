'use client';

import { useEffect, useState, useRef, useCallback } from 'react';
import {
  Line,
  XAxis,
  YAxis,
  CartesianGrid,
  Tooltip,
  ResponsiveContainer,
  Legend,
  Area,
  ComposedChart,
} from 'recharts';
import { predictEos } from '../lib/api';
import { formatKtc, formatKtcTick, clampKtc, KTC_MAX } from '../lib/format';
import type { EOSPrediction, Player } from '../types/player';

const COLORS = ['#2563eb', '#10b981', '#f59e0b', '#ef4444', '#8b5cf6'];
const PPG_RANGE = Array.from({ length: 13 }, (_, i) => i * 2); // 0, 2, 4, ... 24

interface ComparisonPredictionChartProps {
  predictions: EOSPrediction[];
  players: Player[];
  whatIfGames: number;
}

interface ChartDataPoint {
  ppg: number;
  [key: string]: number | undefined;
}

interface CustomTooltipProps {
  active?: boolean;
  payload?: Array<{
    dataKey: string;
    value: number;
    color: string;
    name: string;
  }>;
  label?: number;
}

function CustomTooltip({ active, payload, label }: CustomTooltipProps) {
  if (!active || !payload || !payload.length) return null;

  // Filter out confidence band entries (low/high/band keys)
  const lines = payload.filter(
    (entry) => !entry.dataKey.endsWith('_low') && !entry.dataKey.endsWith('_high') && !entry.dataKey.endsWith('_band')
  );

  return (
    <div className="bg-white dark:bg-gray-800 p-3 rounded-lg shadow-lg border border-gray-200 dark:border-gray-700">
      <p className="text-sm font-medium mb-2 text-gray-900 dark:text-white">
        PPG: {label}
      </p>
      {lines.map((entry, idx) => (
        <p key={idx} className="text-sm" style={{ color: entry.color }}>
          {entry.dataKey}: {formatKtc(entry.value)}
        </p>
      ))}
    </div>
  );
}

export default function ComparisonPredictionChart({
  predictions,
  players,
  whatIfGames,
}: ComparisonPredictionChartProps) {
  const [chartData, setChartData] = useState<ChartDataPoint[]>([]);
  const [loading, setLoading] = useState(false);
  const debounceRef = useRef<ReturnType<typeof setTimeout> | null>(null);
  const predictionsRef = useRef(predictions);
  const playersRef = useRef(players);

  // Keep refs in sync with props
  useEffect(() => {
    predictionsRef.current = predictions;
  }, [predictions]);

  useEffect(() => {
    playersRef.current = players;
  }, [players]);

  const fetchAllPredictions = useCallback(async () => {
    const currentPredictions = predictionsRef.current;
    const currentPlayers = playersRef.current;
    if (currentPredictions.length === 0) return;

    setLoading(true);
    try {
      // For each player, fetch predictions across all PPG values 0-24
      const playerResults = await Promise.all(
        currentPredictions.map(async (pred, idx) => {
          // Get age from player's latest season
          const player = currentPlayers[idx];
          const latestSeason = player?.seasons?.reduce(
            (a, b) => (a.year > b.year ? a : b),
            player.seasons[0]
          );
          const age = latestSeason?.age;

          const results = await Promise.all(
            PPG_RANGE.map((ppg) =>
              predictEos({
                position: pred.position,
                start_ktc: pred.start_ktc,
                games_played: whatIfGames,
                ppg: ppg,
                age: age,
              })
            )
          );
          return { name: pred.name || 'Unknown', results };
        })
      );

      // Build chart data: one point per PPG value
      const data: ChartDataPoint[] = PPG_RANGE.map((ppg, idx) => {
        const point: ChartDataPoint = { ppg };
        playerResults.forEach(({ name, results }) => {
          const result = results[idx];
          if (result) {
            // Clamp all KTC values to valid range [1, 9999]
            point[name] = clampKtc(result.predicted_end_ktc);
            if (result.low_end_ktc != null && result.high_end_ktc != null) {
              const low = clampKtc(result.low_end_ktc);
              const high = clampKtc(result.high_end_ktc);
              point[`${name}_low`] = low;
              point[`${name}_high`] = high;
              point[`${name}_band`] = high - low; // Band height for stacking
            }
          }
        });
        return point;
      });

      setChartData(data);
    } catch (err) {
      console.error('Prediction chart fetch failed:', err);
    } finally {
      setLoading(false);
    }
  }, [whatIfGames, players]);

  useEffect(() => {
    if (predictions.length === 0) return;
    if (debounceRef.current) clearTimeout(debounceRef.current);
    debounceRef.current = setTimeout(fetchAllPredictions, 300);
    return () => {
      if (debounceRef.current) clearTimeout(debounceRef.current);
    };
  }, [fetchAllPredictions, predictions.length, players]);

  if (predictions.length === 0) return null;

  // Don't render chart until data is loaded
  if (chartData.length === 0) {
    return (
      <div className="bg-white dark:bg-gray-800 p-6 rounded-xl shadow-soft border border-gray-100 dark:border-gray-700">
        <h3 className="text-lg font-semibold text-gray-900 dark:text-white mb-2">
          Predicted EOS KTC by PPG
        </h3>
        <div className="h-[350px] flex items-center justify-center">
          <div className="w-6 h-6 border-2 border-gray-200 dark:border-gray-600 border-t-blue-600 rounded-full animate-spin" />
        </div>
      </div>
    );
  }

  // Compute Y-axis domain from chart data
  let minKtc = Infinity;
  let maxKtc = -Infinity;
  chartData.forEach((point) => {
    predictions.forEach((pred) => {
      const name = pred.name || 'Unknown';
      const val = point[name];
      const low = point[`${name}_low`];
      const high = point[`${name}_high`];
      if (typeof val === 'number') {
        minKtc = Math.min(minKtc, val);
        maxKtc = Math.max(maxKtc, val);
      }
      if (typeof low === 'number') minKtc = Math.min(minKtc, low);
      if (typeof high === 'number') maxKtc = Math.max(maxKtc, high);
    });
  });
  if (!isFinite(minKtc)) { minKtc = 0; maxKtc = 10000; }


  return (
    <div className="bg-white dark:bg-gray-800 p-6 rounded-xl shadow-soft border border-gray-100 dark:border-gray-700">
      <h3 className="text-lg font-semibold text-gray-900 dark:text-white mb-2">
        Predicted EOS KTC by PPG
      </h3>
      <p className="text-sm text-gray-500 dark:text-gray-400 mb-4">
        At {whatIfGames} games played — hover to compare values
      </p>

      <div className="relative">
        {loading && (
          <div className="absolute inset-0 bg-white/60 dark:bg-gray-800/60 z-10 flex items-center justify-center rounded-lg">
            <div className="w-6 h-6 border-2 border-gray-200 dark:border-gray-600 border-t-blue-600 rounded-full animate-spin" />
          </div>
        )}

        <ResponsiveContainer width="100%" height={350}>
          <ComposedChart data={chartData} margin={{ top: 20, right: 30, bottom: 35, left: 25 }}>
            <CartesianGrid strokeDasharray="3 3" stroke="#e5e7eb" />
            <XAxis
              dataKey="ppg"
              type="number"
              domain={[0, 25]}
              ticks={[0, 5, 10, 15, 20, 25]}
              tick={{ fontSize: 12 }}
              label={{ value: 'PPG', position: 'bottom', offset: 0, fontSize: 12 }}
            />
            <YAxis
              tickFormatter={formatKtcTick}
              domain={[0, KTC_MAX]}
              ticks={[0, 2500, 5000, 7500, KTC_MAX]}
              tick={{ fontSize: 12 }}
              width={60}
              label={{ value: 'Predicted EOS KTC', angle: -90, position: 'insideLeft', fontSize: 12, dx: -5 }}
            />
            <Tooltip content={<CustomTooltip />} />
            <Legend
              formatter={(value: string) =>
                value.endsWith('_low') || value.endsWith('_high') || value.endsWith('_band') ? '' : value
              }
            />

            {/* Confidence bands as stacked areas (low + band height) */}
            {predictions.map((pred, idx) => {
              const name = pred.name || 'Unknown';
              const hasConfidence = chartData.some(
                (d) => d[`${name}_low`] != null && d[`${name}_high`] != null
              );
              if (!hasConfidence) return null;
              const color = COLORS[idx % COLORS.length];
              const stackId = `band-${pred.player_id}`;
              return [
                // Invisible base area from 0 to low
                <Area
                  key={`band-base-${pred.player_id}`}
                  dataKey={`${name}_low`}
                  stackId={stackId}
                  stroke="none"
                  fill="transparent"
                  legendType="none"
                  tooltipType="none"
                  isAnimationActive={false}
                />,
                // Visible band area from low to high
                <Area
                  key={`band-fill-${pred.player_id}`}
                  dataKey={`${name}_band`}
                  stackId={stackId}
                  stroke="none"
                  fill={color}
                  fillOpacity={0.15}
                  legendType="none"
                  tooltipType="none"
                  isAnimationActive={false}
                />,
              ];
            })}

            {/* Main prediction lines */}
            {predictions.map((pred, idx) => (
              <Line
                key={pred.player_id || idx}
                type="monotone"
                dataKey={pred.name || 'Unknown'}
                stroke={COLORS[idx % COLORS.length]}
                strokeWidth={2}
                dot={false}
                activeDot={{ r: 5, fill: COLORS[idx % COLORS.length], stroke: '#fff', strokeWidth: 2 }}
              />
            ))}

          </ComposedChart>
        </ResponsiveContainer>
      </div>
    </div>
  );
}
